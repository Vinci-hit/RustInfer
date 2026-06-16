//! Token-granularity RadixTree for prefix reuse over worker-owned KV slots.
//!
//! The worker owns the physical KV pool and its `GlobalKvAllocator`. This
//! `RadixTree` is the scheduler's prefix-cache index: nodes hold **handles**
//! to per-token slots in the worker's pool — `Vec<u32>` of global token
//! indices. Nodes are inserted incrementally one token at a time as the
//! worker reports each `StepOutput.assigned_indices`. Eviction yields a
//! `Vec<u32>` of indices that the scheduler then asks the worker to free
//! via `FreeKvIndices`.
//!
//! ## Reference counting + LRU invariant
//!
//! Every node carries `owners: HashSet<SeqId>` — the set of live sequences
//! whose chain currently walks through this node. `owners.is_empty()` is the
//! single source of truth for "no live decoder pins this node".
//!
//! **Core invariant**: a node is in the LRU list **iff** `owners.is_empty()`
//! **and** it is a leaf of its current subtree. This single rule keeps the
//! eviction set free of any token slot a live request could legitimately
//! point to:
//!
//! - When `lookup_prefix` matches a node along a previously-unowned chain,
//!   the new sequence is added to every matched node's `owners`. The nodes
//!   are removed from the LRU list in the same call. Subsequent `evict`
//!   calls cannot pick them up.
//! - When `mark_finished_chain` runs, it walks from the chain's leaf back to
//!   the root, removing the seq from each `owners`. Any node whose owners
//!   set becomes empty *and* has no still-owned children joins the LRU tail.
//!
//! The scheduler's main loop is single-threaded, so `lookup_prefix → evict →
//! send step` runs serially within one iteration. There is no race between
//! a prefix hit and a concurrent eviction.
//!
//! ## Edge granularity
//!
//! Edges are *token sequences*; new tokens append one at a time. Edge length
//! threshold (`EDGE_SPLIT_THRESHOLD`) controls when we split a long edge so
//! a divergent sibling can branch off mid-edge. `append_token` always grows
//! the leaf along the seq's current chain; splits happen lazily inside
//! `lookup_prefix` when a sibling's prefix forces a split.

use std::collections::{HashMap, HashSet, VecDeque};

/// Stable sequence identifier. Aliased to `u64` to match
/// `crate::domain::inference_session::lifecycle::SequenceId`'s inner type.
pub type SeqId = u64;

/// Worker-side global token-slot index.
pub type GlobalIndex = u32;

/// Threshold above which a single tree edge gets split when a divergent
/// sibling appears. Token-by-token append lets us keep this large; we only
/// pay for splits when prefixes actually fork. Chosen liberally; tunable.
const EDGE_SPLIT_THRESHOLD: usize = 32;

#[derive(Debug)]
struct Node {
    /// Tokens labeling the edge from this node's parent to itself. Empty for
    /// the synthetic root.
    edge_tokens: Vec<i32>,
    /// Global KV indices for the tokens in `edge_tokens`. Always
    /// `edge_tokens.len() == global_indices.len()`.
    global_indices: Vec<GlobalIndex>,
    parent: Option<NodeId>,
    /// First-token → child node id. We key on the first edge token because
    /// each edge can only branch on its first token (standard radix invariant).
    children: HashMap<i32, NodeId>,
    owners: HashSet<SeqId>,
    /// Whether this node currently sits in `lru.queue`. Mirrors the
    /// `owners.is_empty() + leaf` invariant for O(1) check; we still verify
    /// on pop because removal is lazy (see `LruList::remove`).
    in_lru: bool,
}

type NodeId = usize;

/// LRU queue. We use a VecDeque + a generation stamp so removals can be lazy
/// (we just bump the generation; stale entries are skipped at pop time).
#[derive(Debug, Default)]
struct LruList {
    queue: VecDeque<(NodeId, u64)>,
    /// Per-node "current generation". Entries in `queue` whose stored gen
    /// disagrees with this are stale and skipped on pop. Using a HashMap
    /// keeps memory proportional to *currently-tracked* nodes.
    generations: HashMap<NodeId, u64>,
}

impl LruList {
    fn push_tail(&mut self, node: NodeId) {
        let g = self.generations.entry(node).or_insert(0);
        *g = g.wrapping_add(1);
        self.queue.push_back((node, *g));
    }

    /// Mark the entry stale. Caller still owns whether the node is "in lru"
    /// for invariant purposes — this just makes `pop` skip it. We rely on
    /// the caller to update `Node.in_lru`.
    fn remove(&mut self, node: NodeId) {
        if let Some(g) = self.generations.get_mut(&node) {
            *g = g.wrapping_add(1);
        }
    }

    fn pop_front_valid(&mut self) -> Option<NodeId> {
        while let Some((node, g)) = self.queue.pop_front() {
            if let Some(&cur) = self.generations.get(&node) {
                if cur == g {
                    return Some(node);
                }
            }
        }
        None
    }

    fn len_estimate(&self) -> usize {
        self.queue.len()
    }
}

/// Per-sequence chain pointer: the leaf node of that seq's current chain.
/// `append_token` walks/extends from this pointer; `mark_finished_chain`
/// walks from this pointer up to the root.
#[derive(Debug, Default, Clone, Copy)]
struct ChainTip {
    leaf: NodeId,
    /// Position within `leaf.edge_tokens` of the seq's last token. The seq
    /// "owns" `leaf.edge_tokens[..=pos]`. Multiple seqs sharing the same
    /// leaf node may have different `pos` values when one is mid-edge and
    /// the other has consumed the full edge — but in practice we split
    /// edges on divergence so this stays simple.
    pos: usize,
}

/// Token-granularity RadixTree. Single-threaded: the scheduler event loop
/// owns the only `&mut`.
#[derive(Debug)]
pub struct RadixTree {
    nodes: Vec<Node>,
    /// Synthetic root, always `nodes[0]`.
    root: NodeId,
    lru: LruList,
    /// Per-seq chain tip. Inserted on first `append_token` or
    /// `lookup_prefix`; removed when `mark_finished_chain` clears the seq.
    seqs: HashMap<SeqId, ChainTip>,
    /// Tombstoned node slots, free for reuse by `new_node` (H5). A node lands
    /// here when `evict` logically deletes it (no owners, no children).
    free_ids: Vec<NodeId>,
}

/// Outcome of `lookup_prefix`. `matched_indices` is the flat list of global
/// indices the new request can reuse (in order); `pinned_nodes` is the set
/// of nodes whose owners now include the new seq, exposed for callers that
/// want to track per-step temporary holds separately.
#[derive(Debug, Clone)]
pub struct PrefixHit {
    pub matched_indices: Vec<GlobalIndex>,
    pub pinned_nodes: Vec<NodeId>,
}

impl RadixTree {
    pub fn new() -> Self {
        let root = Node {
            edge_tokens: Vec::new(),
            global_indices: Vec::new(),
            parent: None,
            children: HashMap::new(),
            owners: HashSet::new(),
            // Root is always pinned; never in LRU.
            in_lru: false,
        };
        Self {
            nodes: vec![root],
            root: 0,
            lru: LruList::default(),
            seqs: HashMap::new(),
            free_ids: Vec::new(),
        }
    }

    /// Allocate a node slot, reusing a tombstoned `NodeId` when one is
    /// available (H5). Without reuse, `evict` only logically clears nodes
    /// (id stability matters for `ChainTip`/`lru.generations`) so `nodes`
    /// grew monotonically on long runs. The free-list bounds `nodes.len()`
    /// to the *peak* live tree size instead. Reused ids keep their (stale,
    /// higher) `lru.generations` stamp, so any lingering stale queue entry
    /// for that id is still correctly skipped at pop time.
    fn new_node(&mut self, node: Node) -> NodeId {
        if let Some(id) = self.free_ids.pop() {
            self.nodes[id] = node;
            id
        } else {
            let id = self.nodes.len();
            self.nodes.push(node);
            id
        }
    }

    /// Number of tokens currently held in the tree (sum across all nodes).
    /// Pure debug accessor; not on hot path.
    pub fn token_count(&self) -> usize {
        self.nodes
            .iter()
            .skip(1) // root has empty edge
            .map(|n| n.edge_tokens.len())
            .sum()
    }

    /// LRU queue length estimate (may include stale entries; pops skip them).
    pub fn lru_len_estimate(&self) -> usize {
        self.lru.len_estimate()
    }

    /// True if `seq_id` has a live chain in the tree.
    pub fn contains_seq(&self, seq_id: SeqId) -> bool {
        self.seqs.contains_key(&seq_id)
    }

    /// Append one new token to `seq_id`'s chain end. Creates the seq's chain
    /// pointer at the root on first call.
    ///
    /// Worker-side `assigned_indices` from one StepOutput drive a sequence
    /// of `append_token` calls; the scheduler handles them serially.
    pub fn append_token(&mut self, seq_id: SeqId, token_id: i32, global_idx: GlobalIndex) {
        // Snapshot tip; we'll write back at the end. Avoids holding a
        // mutable borrow on `self.seqs` across calls into `self.split_edge`
        // / `self.add_owner`.
        let mut tip = *self.seqs.entry(seq_id).or_insert(ChainTip {
            leaf: self.root,
            pos: 0,
        });

        // Case A: still inside the leaf's edge — try to consume one more
        // edge token if it matches the incoming token. This happens only
        // for owners holding a sub-prefix of the leaf's edge (rare in
        // practice, but legal).
        if tip.leaf != self.root && tip.pos < self.nodes[tip.leaf].edge_tokens.len() {
            if self.nodes[tip.leaf].edge_tokens[tip.pos] == token_id {
                debug_assert_eq!(
                    self.nodes[tip.leaf].global_indices[tip.pos], global_idx,
                    "global index drift mid-edge: seq {} pos {} expected {}, got {}",
                    seq_id, tip.pos, self.nodes[tip.leaf].global_indices[tip.pos], global_idx,
                );
                tip.pos += 1;
                self.seqs.insert(seq_id, tip);
                return;
            }
            // Diverging mid-edge: split the leaf at `tip.pos`.
            self.split_edge(tip.leaf, tip.pos);
            // After split, the original leaf's edge length equals tip.pos.
            // Re-read the tip from `seqs` because split_edge may have
            // updated owners whose pos > split point.
            tip = self.seqs[&seq_id];
            // We're now at the original leaf's boundary.
            tip.pos = self.nodes[tip.leaf].edge_tokens.len();
        }

        // Case B: at the boundary of `leaf`. Look for a child whose first
        // edge token matches.
        if let Some(&child) = self.nodes[tip.leaf].children.get(&token_id) {
            self.add_owner(child, seq_id);
            let edge_len = self.nodes[child].edge_tokens.len();
            debug_assert!(edge_len >= 1);
            debug_assert_eq!(self.nodes[child].edge_tokens[0], token_id);
            debug_assert_eq!(self.nodes[child].global_indices[0], global_idx);
            tip.leaf = child;
            tip.pos = 1;
            self.seqs.insert(seq_id, tip);
            return;
        }

        // Case C: no matching child — extend by either growing the current
        // leaf's edge or by creating a new leaf.
        if self.can_grow_leaf_in_place(tip.leaf, seq_id) {
            self.nodes[tip.leaf].edge_tokens.push(token_id);
            self.nodes[tip.leaf].global_indices.push(global_idx);
            tip.pos = self.nodes[tip.leaf].edge_tokens.len();
            self.seqs.insert(seq_id, tip);
            return;
        }

        // Create a new child node.
        let parent = tip.leaf;
        let new_id = self.new_node(Node {
            edge_tokens: vec![token_id],
            global_indices: vec![global_idx],
            parent: Some(parent),
            children: HashMap::new(),
            owners: {
                let mut s = HashSet::new();
                s.insert(seq_id);
                s
            },
            in_lru: false,
        });
        self.nodes[parent].children.insert(token_id, new_id);
        // Adding a child to `parent` may have un-leafed it.
        self.demote_from_lru_if_present(parent);
        tip.leaf = new_id;
        tip.pos = 1;
        self.seqs.insert(seq_id, tip);
    }

    /// Look up the longest prefix match for `tokens`, attaching `new_seq_id`
    /// to every matched node along the way (so they pin and leave LRU).
    ///
    /// Caller must call `append_token` for every prompt token *not* covered
    /// by `matched_indices`. Worker side will see `prefix_hint =
    /// Some(matched_indices)` in the prefill segment and skip those tokens.
    pub fn lookup_prefix(&mut self, tokens: &[i32], new_seq_id: SeqId) -> PrefixHit {
        let mut node = self.root;
        let mut pos = 0usize;
        let mut matched: Vec<GlobalIndex> = Vec::new();
        let mut pinned: Vec<NodeId> = Vec::new();
        let mut tok_idx = 0usize;

        while tok_idx < tokens.len() {
            // If we're at a node boundary, try to descend into a child.
            if pos == self.nodes[node].edge_tokens.len() {
                let next = match self.nodes[node].children.get(&tokens[tok_idx]) {
                    Some(&c) => c,
                    None => break,
                };
                node = next;
                pos = 0;
                continue;
            }

            // We're partway through `node.edge_tokens`. Match next token.
            if self.nodes[node].edge_tokens[pos] != tokens[tok_idx] {
                // Mismatch mid-edge: matched up to `pos`. We do NOT split here;
                // we just stop. The caller will start `append_token` at the
                // current chain tip when it later sees fresh tokens, and the
                // append path will handle splits.
                break;
            }
            matched.push(self.nodes[node].global_indices[pos]);
            tok_idx += 1;
            pos += 1;
        }

        // Attach `new_seq_id` to every node we landed on (root excluded).
        // Walk back from `node` to root.
        let mut cur = node;
        while cur != self.root {
            self.add_owner(cur, new_seq_id);
            pinned.push(cur);
            cur = self.nodes[cur].parent.expect("non-root has parent");
        }
        pinned.reverse(); // root → leaf order

        // Place the new seq's tip at (node, pos) so subsequent append_token
        // continues from where lookup ended.
        if !matched.is_empty() {
            self.seqs.insert(new_seq_id, ChainTip { leaf: node, pos });
        } else {
            // No prefix hit at all — fresh seq starts at root.
            self.seqs.insert(
                new_seq_id,
                ChainTip {
                    leaf: self.root,
                    pos: 0,
                },
            );
        }

        PrefixHit {
            matched_indices: matched,
            pinned_nodes: pinned,
        }
    }

    /// Mark `seq_id`'s entire chain as no longer owned by this seq. Nodes
    /// whose owner set becomes empty (and have no live descendants) enter
    /// the LRU tail.
    ///
    /// Idempotent: calling on an unknown / already-finished seq is a no-op.
    /// A `StepOutput` arriving after cancel must still be appendable; this
    /// models the post-StepOutput cleanup path.
    pub fn mark_finished_chain(&mut self, seq_id: SeqId) {
        let Some(ChainTip { leaf, pos: _ }) = self.seqs.remove(&seq_id) else {
            return;
        };
        let mut cur = leaf;
        while cur != self.root {
            self.nodes[cur].owners.remove(&seq_id);
            if self.nodes[cur].owners.is_empty() {
                // Only insert into LRU if currently a leaf in the tree.
                if self.nodes[cur].children.is_empty() && !self.nodes[cur].in_lru {
                    self.nodes[cur].in_lru = true;
                    self.lru.push_tail(cur);
                }
            }
            cur = self.nodes[cur].parent.expect("non-root has parent");
        }
    }

    /// Sum of `global_indices.len()` across every node that can be
    /// reclaimed by repeated LRU eviction from the current tree state.
    ///
    /// The physical LRU queue only contains unowned leaves. Evicting a
    /// leaf can turn its unowned parent into the next LRU leaf, so admission
    /// must count the whole reclaimable subtree, not just nodes already
    /// present in `lru.queue`. Otherwise long finished chains look like a
    /// few dozen freeable slots and the scheduler can starve waiting work
    /// even though `evict_collect_at_least` could free enough KV.
    pub fn lru_total_indices(&self) -> usize {
        self.nodes[self.root]
            .children
            .values()
            .copied()
            .map(|child| self.reclaimable_subtree_total(child).0)
            .sum()
    }

    /// Evict from the LRU front until at least `target_n` global
    /// indices are gathered. Atomic at the node level — a popped
    /// node's full `global_indices` always ships, never a slice. May
    /// return *more* than `target_n` (the "at_least" in the name) and
    /// *less* if the LRU drains before reaching the target. Caller
    /// must escalate to victim preemption when the returned count is
    /// short of what they need.
    pub fn evict_collect_at_least(&mut self, target_n: usize) -> Vec<GlobalIndex> {
        let mut out: Vec<GlobalIndex> = Vec::new();
        while out.len() < target_n {
            let Some(node_id) = self.lru.pop_front_valid() else {
                break;
            };
            // The node may have been re-pinned between push and pop (lookup
            // raised owners 0→1) — skip in that case.
            if self.nodes[node_id].owners.is_empty() && self.nodes[node_id].children.is_empty() {
                self.nodes[node_id].in_lru = false;
                let parent = self.nodes[node_id]
                    .parent
                    .expect("non-root cannot be in LRU");
                // Detach from parent.
                let first_tok = self.nodes[node_id].edge_tokens[0];
                self.nodes[parent].children.remove(&first_tok);
                // Gather its global indices.
                out.extend_from_slice(&self.nodes[node_id].global_indices);
                // Logical delete: clear the node so any stale refs see empty
                // arrays. We do not compact `nodes` (id stability matters
                // for ChainTip and lru.gen).
                self.nodes[node_id].edge_tokens.clear();
                self.nodes[node_id].global_indices.clear();
                self.nodes[node_id].parent = None;
                self.nodes[node_id].owners.clear();
                // Reclaim the slot for reuse (H5). `children` is already empty
                // (eviction precondition). The id keeps its monotonic
                // `lru.generations` stamp so stale queue entries stay invalid.
                self.free_ids.push(node_id);
                // Removing this child may have made the parent an unowned
                // leaf — promote it to LRU.
                self.maybe_promote_to_lru(parent);
            } else {
                // Stale entry; just drop and keep popping.
                self.nodes[node_id].in_lru = false;
            }
        }
        out
    }

    /// Backwards-compat shim. `evict_collect_at_least` is the new
    /// canonical name. Existing tests and callers that already spelled
    /// the old name keep working until they're migrated.
    #[inline]
    pub fn evict(&mut self, target_n: usize) -> Vec<GlobalIndex> {
        self.evict_collect_at_least(target_n)
    }

    // ─── Internal helpers ─────────────────────────────────────────────

    /// Add `seq_id` to `node.owners`. If the node was previously unowned and
    /// in LRU, take it out: it now has a live owner.
    fn add_owner(&mut self, node: NodeId, seq_id: SeqId) {
        let was_empty = self.nodes[node].owners.is_empty();
        self.nodes[node].owners.insert(seq_id);
        if was_empty && self.nodes[node].in_lru {
            self.lru.remove(node);
            self.nodes[node].in_lru = false;
        }
    }

    /// True iff we can grow the leaf's edge in place: leaf is exclusively
    /// owned by this seq, has no other children, and is below the split
    /// threshold. This keeps single-seq decode-only chains compact.
    fn can_grow_leaf_in_place(&self, leaf: NodeId, seq_id: SeqId) -> bool {
        if leaf == self.root {
            return false;
        }
        let n = &self.nodes[leaf];
        n.children.is_empty()
            && n.owners.len() == 1
            && n.owners.contains(&seq_id)
            && n.edge_tokens.len() < EDGE_SPLIT_THRESHOLD
    }

    /// Split a node's edge at position `pos` (0 < pos < edge_tokens.len()).
    /// The original node retains its first `pos` tokens; a new suffix child
    /// is created holding the suffix and inheriting the original node's
    /// children. Owner placement after the split:
    ///
    /// - Owners whose `tip.pos > pos` were past the split → migrate to the
    ///   suffix node (and only the suffix). Their tip is updated.
    /// - Owners whose `tip.pos <= pos` were at or before the split → stay
    ///   only on the prefix node.
    /// - Children inherited by the suffix node bring with them owners whose
    ///   chains ran through them: those owners must already appear in the
    ///   suffix's `owners` because we cloned the original owners and then
    ///   *kept on the prefix* the ones that were not past the split. So the
    ///   suffix's owner set = (original owners with tip.pos > pos) ∪ (any
    ///   transitive owner of children we inherited — already in original
    ///   owners by our path-ownership invariant).
    fn split_edge(&mut self, node: NodeId, pos: usize) {
        debug_assert!(node != self.root);
        debug_assert!(pos > 0 && pos < self.nodes[node].edge_tokens.len());

        let tail_tokens = self.nodes[node].edge_tokens[pos..].to_vec();
        let tail_indices = self.nodes[node].global_indices[pos..].to_vec();
        let original_children = std::mem::take(&mut self.nodes[node].children);

        // Compute the partition of owners: those past `pos` migrate to
        // suffix, those at-or-before stay on prefix.
        let owners_past: HashSet<SeqId> = self
            .seqs
            .iter()
            .filter(|(_, t)| t.leaf == node && t.pos > pos)
            .map(|(&s, _)| s)
            .collect();
        // Owners that should remain on the prefix = original.owners − owners_past
        // … but ALSO owners whose chains run *through* this subtree via
        // inherited children must be on suffix. Because of path-ownership,
        // any such owner is already in `node.owners` (their chain passes
        // through the original node), and they must appear on both prefix
        // *and* suffix after the split. So the rule simplifies to:
        //   suffix_owners = original.owners (everything that walked through)
        //   prefix_owners = original.owners (still walks through prefix)
        // EXCEPT owners whose tip is exactly at the prefix boundary (pos)
        // and have no children below them — those owners do NOT walk
        // through the suffix and should be removed from the suffix's owner
        // set. We compute this by walking children of the suffix and
        // collecting the union of their owner sets. Owners absent from
        // that union but present in the original owner set with tip at
        // (node, pos) belong only to the prefix.
        let suffix_inherited_children: Vec<NodeId> = original_children.values().copied().collect();
        let mut suffix_transitive_owners: HashSet<SeqId> = HashSet::new();
        for c in &suffix_inherited_children {
            for &o in self.nodes[*c].owners.iter() {
                suffix_transitive_owners.insert(o);
            }
        }
        // suffix_owners = owners_past ∪ suffix_transitive_owners
        let mut suffix_owners = owners_past.clone();
        for o in &suffix_transitive_owners {
            suffix_owners.insert(*o);
        }
        // prefix_owners = original.owners − (those whose chain ends
        // exactly at this node and didn't go past). i.e. keep an owner on
        // the prefix iff it is past the split (tip > pos), OR it is in
        // suffix_transitive_owners (chain runs through suffix), OR its
        // tip is at this node with pos <= split_point (still pinned on
        // prefix's range).
        // Simpler: prefix_owners = self.nodes[node].owners (unchanged).
        // The original owner set already represents "chains that walked
        // through this edge"; truncating the edge does not invalidate any
        // of those.
        let prefix_owners = self.nodes[node].owners.clone();

        // Truncate the original node to the prefix.
        self.nodes[node].edge_tokens.truncate(pos);
        self.nodes[node].global_indices.truncate(pos);
        // children was already drained via mem::take above.

        // Create the suffix node.
        let first_tok = tail_tokens[0];
        let suffix_id = self.new_node(Node {
            edge_tokens: tail_tokens,
            global_indices: tail_indices,
            parent: Some(node),
            children: original_children,
            owners: suffix_owners,
            in_lru: false,
        });

        // Relink inherited children's parent.
        for c in suffix_inherited_children {
            self.nodes[c].parent = Some(suffix_id);
        }
        self.nodes[node].children.insert(first_tok, suffix_id);

        // Restore prefix owners (we left them unchanged but be explicit).
        self.nodes[node].owners = prefix_owners;

        // Move owners whose tip lived inside the suffix down to the suffix.
        let mut to_update = Vec::new();
        for (sid, tip) in self.seqs.iter() {
            if tip.leaf == node && tip.pos > pos {
                to_update.push((*sid, tip.pos - pos));
            }
        }
        for (sid, new_pos) in to_update {
            let t = self.seqs.get_mut(&sid).unwrap();
            t.leaf = suffix_id;
            t.pos = new_pos;
        }

        // The original node now has the suffix as a child, so it's not a
        // leaf any more — it cannot remain in LRU.
        self.demote_from_lru_if_present(node);

        // The suffix node may need LRU promotion if it's an unowned leaf
        // (it has no children only when original had no children, which is
        // false when we created the split because original had its edge
        // continue past pos — so suffix has no inherited children only
        // when original was already a leaf). Easier: if suffix has no
        // owners and no children → maybe_promote.
        self.maybe_promote_to_lru(suffix_id);
    }

    /// If a node is currently in the LRU and an event makes it ineligible
    /// (e.g. it gained a child), kick it out.
    fn demote_from_lru_if_present(&mut self, node: NodeId) {
        if self.nodes[node].in_lru {
            self.lru.remove(node);
            self.nodes[node].in_lru = false;
        }
    }

    /// If a node has just become an unowned leaf (no owners + no children),
    /// promote it into the LRU.
    fn maybe_promote_to_lru(&mut self, node: NodeId) {
        if node == self.root {
            return;
        }
        let n = &self.nodes[node];
        if !n.in_lru && n.owners.is_empty() && n.children.is_empty() && !n.edge_tokens.is_empty()
        // not a logically-deleted node
        {
            self.nodes[node].in_lru = true;
            self.lru.push_tail(node);
        }
    }

    fn reclaimable_subtree_total(&self, node: NodeId) -> (usize, bool) {
        let mut total = 0usize;
        let mut children_fully_reclaimable = true;

        for &child in self.nodes[node].children.values() {
            let (child_total, child_fully_reclaimable) = self.reclaimable_subtree_total(child);
            total += child_total;
            children_fully_reclaimable &= child_fully_reclaimable;
        }

        let fully_reclaimable = self.nodes[node].owners.is_empty() && children_fully_reclaimable;
        if fully_reclaimable {
            total += self.nodes[node].global_indices.len();
        }

        (total, fully_reclaimable)
    }
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn append_seq(tree: &mut RadixTree, seq: SeqId, tokens: &[(i32, GlobalIndex)]) {
        for &(t, idx) in tokens {
            tree.append_token(seq, t, idx);
        }
    }

    #[test]
    fn evicted_node_ids_are_reused_bounding_node_vec() {
        // H5: tombstoned slots must be recycled so `nodes` is bounded by the
        // peak live tree size, not the cumulative number of nodes ever made.
        let mut t = RadixTree::new();
        append_seq(&mut t, 1, &[(10, 100), (20, 101)]);
        t.mark_finished_chain(1);
        let peak = t.nodes.len();
        let evicted = t.evict(100);
        assert!(!evicted.is_empty());
        assert!(
            !t.free_ids.is_empty(),
            "evicted nodes must land on the free list"
        );
        assert_eq!(t.nodes.len(), peak, "tombstones do not shrink nodes vec");

        // Repeatedly build + finish + evict equivalent chains. With reuse the
        // node vector never grows past the first peak.
        for s in 2..50u64 {
            append_seq(&mut t, s, &[(10, 100), (20, 101)]);
            t.mark_finished_chain(s);
            let _ = t.evict(100);
            assert!(
                t.nodes.len() <= peak,
                "node ids not reused: nodes grew to {} (peak {})",
                t.nodes.len(),
                peak
            );
        }

        // Tree still functions correctly after heavy reuse.
        append_seq(&mut t, 100, &[(30, 200), (40, 201)]);
        let hit = t.lookup_prefix(&[30, 40], 101);
        assert_eq!(hit.matched_indices, vec![200, 201]);
    }

    #[test]
    fn empty_tree_has_only_root() {
        let t = RadixTree::new();
        assert_eq!(t.token_count(), 0);
        assert_eq!(t.lru_len_estimate(), 0);
    }

    #[test]
    fn append_grows_in_place_on_single_owner() {
        let mut t = RadixTree::new();
        append_seq(&mut t, 1, &[(10, 100), (20, 101), (30, 102)]);
        // All three tokens collapse into a single edge from root.
        assert_eq!(t.token_count(), 3);
        assert!(t.contains_seq(1));
    }

    #[test]
    fn lookup_no_match_returns_empty() {
        let mut t = RadixTree::new();
        append_seq(&mut t, 1, &[(10, 100), (20, 101)]);
        // Different token at start: zero match.
        let hit = t.lookup_prefix(&[99], 2);
        assert!(hit.matched_indices.is_empty());
        assert!(t.contains_seq(2));
    }

    #[test]
    fn lookup_partial_match_returns_indices_and_pins_nodes() {
        let mut t = RadixTree::new();
        append_seq(&mut t, 1, &[(10, 100), (20, 101), (30, 102)]);
        // seq 1 finishes → its chain becomes Finished + enters LRU.
        t.mark_finished_chain(1);
        assert!(t.lru_len_estimate() >= 1);
        // New seq 2 looks up the same prefix.
        let hit = t.lookup_prefix(&[10, 20, 30, 40], 2);
        assert_eq!(hit.matched_indices, vec![100, 101, 102]);
        // Pinning lifts the chain out of LRU. evict(target=10) returns nothing
        // because the chain is now pinned by seq 2.
        let evicted = t.evict(10);
        assert!(
            evicted.is_empty(),
            "pinned nodes must not evict, got {:?}",
            evicted
        );
    }

    #[test]
    fn finished_chain_evicts_to_indices() {
        let mut t = RadixTree::new();
        append_seq(&mut t, 1, &[(10, 100), (20, 101), (30, 102)]);
        t.mark_finished_chain(1);
        let evicted = t.evict(3);
        assert_eq!(evicted.len(), 3);
        let mut sorted = evicted.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![100, 101, 102]);
    }

    #[test]
    fn shared_prefix_two_seqs_one_finishes_keeps_other_pinned() {
        let mut t = RadixTree::new();
        // Seq 1 writes "A B C".
        append_seq(&mut t, 1, &[(10, 100), (20, 101), (30, 102)]);
        // Seq 2 looks up "A B" prefix (gets indices 100, 101) and continues
        // with a divergent token "X" (writes index 200).
        let hit = t.lookup_prefix(&[10, 20], 2);
        assert_eq!(hit.matched_indices, vec![100, 101]);
        t.append_token(2, 99, 200); // diverge

        // Seq 1 finishes. Its tail "C" (at index 102) should NOT enter LRU
        // immediately because the parent edge "A B" is still owned by seq 2.
        t.mark_finished_chain(1);

        // Eviction should only yield index 102 (the part exclusive to seq 1).
        let evicted = t.evict(10);
        assert_eq!(evicted, vec![102], "evicted: {:?}", evicted);

        // Seq 2 still holds the rest. Looking up its prefix still hits.
        let hit = t.lookup_prefix(&[10, 20], 99);
        assert_eq!(hit.matched_indices, vec![100, 101]);
    }

    #[test]
    fn cancel_already_finished_seq_is_noop() {
        let mut t = RadixTree::new();
        append_seq(&mut t, 1, &[(10, 100)]);
        t.mark_finished_chain(1);
        // Second call must not panic and must not change LRU.
        let lru_before = t.lru_len_estimate();
        t.mark_finished_chain(1);
        assert_eq!(t.lru_len_estimate(), lru_before);
    }

    #[test]
    fn append_token_after_finished_keeps_chain_finished() {
        // Plan §R4: a StepOutput may arrive for a seq after cancel; the
        // RadixTree must accept the late token without re-Decoding the chain.
        let mut t = RadixTree::new();
        append_seq(&mut t, 1, &[(10, 100), (20, 101)]);
        t.mark_finished_chain(1);
        // After mark_finished, seq 1 has no chain tip. A late append_token
        // creates a fresh chain at root — it does NOT resurrect the old
        // chain. The old indices stay in LRU and the new token sits on a
        // brand-new branch (which goes Finished as soon as we mark it).
        t.append_token(1, 30, 102);
        // Old chain's indices still evictable.
        let evicted_before_finishing = t.evict(2);
        let mut sorted = evicted_before_finishing.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![100, 101]);
        // New chain at root → finishing it puts 102 into LRU.
        t.mark_finished_chain(1);
        let evicted = t.evict(1);
        assert_eq!(evicted, vec![102]);
    }

    #[test]
    fn evict_returns_fewer_when_lru_exhausted() {
        let mut t = RadixTree::new();
        append_seq(&mut t, 1, &[(10, 100), (20, 101)]);
        t.mark_finished_chain(1);
        let evicted = t.evict(100);
        assert_eq!(evicted.len(), 2); // only 2 indices ever existed
        let evicted2 = t.evict(1);
        assert!(evicted2.is_empty());
    }

    #[test]
    fn lookup_then_finish_then_evict_full_round_trip() {
        let mut t = RadixTree::new();
        append_seq(&mut t, 1, &[(10, 100), (20, 101), (30, 102)]);
        t.mark_finished_chain(1);
        // Pin via lookup.
        let _ = t.lookup_prefix(&[10, 20, 30], 2);
        let evicted = t.evict(10);
        assert!(evicted.is_empty(), "pinned chain must not evict");
        // Seq 2 finishes — chain returns to LRU.
        t.mark_finished_chain(2);
        let evicted = t.evict(10);
        assert_eq!(evicted.len(), 3);
    }

    #[test]
    fn lookup_full_match_returns_all_indices() {
        let mut t = RadixTree::new();
        append_seq(&mut t, 1, &[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]);
        t.mark_finished_chain(1);
        let hit = t.lookup_prefix(&[1, 2, 3, 4, 5], 99);
        assert_eq!(hit.matched_indices, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn divergent_append_after_lookup_creates_branch_correctly() {
        let mut t = RadixTree::new();
        append_seq(&mut t, 1, &[(10, 100), (20, 101)]);
        t.mark_finished_chain(1);

        // New seq matches "10, 20" then diverges with token 99 / index 200.
        let hit = t.lookup_prefix(&[10, 20], 2);
        assert_eq!(hit.matched_indices, vec![100, 101]);
        t.append_token(2, 99, 200);

        // Now another seq looks up the original [10, 20] — must still work.
        let hit3 = t.lookup_prefix(&[10, 20], 3);
        assert_eq!(hit3.matched_indices, vec![100, 101]);
    }

    // ─── Refcount-only LRU semantics ─────────────────────────────────────

    #[test]
    fn two_seqs_pin_shared_prefix_until_both_finish() {
        // Two sequences walk through the same prefix [10, 20]. While either
        // is still alive, no part of that prefix may be evicted.
        let mut t = RadixTree::new();
        append_seq(&mut t, 1, &[(10, 100), (20, 101), (30, 102)]);

        // Seq 2 reuses [10, 20] then diverges with its own tail [40, 200].
        let hit = t.lookup_prefix(&[10, 20], 2);
        assert_eq!(hit.matched_indices, vec![100, 101]);
        t.append_token(2, 40, 200);

        // Seq 1 finishes: only its private suffix index 102 is releasable.
        t.mark_finished_chain(1);
        let evicted = t.evict(10);
        assert_eq!(
            evicted,
            vec![102],
            "shared prefix must remain pinned by seq 2"
        );

        // Indices 100, 101 are still pinned by seq 2 — also still reusable.
        let hit_again = t.lookup_prefix(&[10, 20], 3);
        assert_eq!(hit_again.matched_indices, vec![100, 101]);

        // Both seq 2 & 3 finish — now the shared prefix can drain.
        t.mark_finished_chain(2);
        t.mark_finished_chain(3);
        let mut evicted2 = t.evict(10);
        evicted2.sort_unstable();
        // seq 2's tail (200) and the shared prefix (100, 101) both surface.
        assert_eq!(evicted2, vec![100, 101, 200]);
    }

    #[test]
    fn split_edge_keeps_prefix_only_owner_out_of_suffix() {
        let mut t = RadixTree::new();
        append_seq(&mut t, 1, &[(10, 100), (20, 101), (30, 102), (40, 103)]);

        // Seq 2 pins only the prefix [10, 20], then diverges. This forces a
        // split of seq 1's compact edge at pos=2.
        let hit = t.lookup_prefix(&[10, 20], 2);
        assert_eq!(hit.matched_indices, vec![100, 101]);
        t.append_token(2, 99, 200);

        // Finishing seq 2 may release only its private divergent suffix. The
        // [30, 40] suffix is still owned by seq 1 and must not be evictable.
        t.mark_finished_chain(2);
        let evicted = t.evict(10);
        assert_eq!(evicted, vec![200]);

        // Seq 1 still pins the original full chain.
        let hit_again = t.lookup_prefix(&[10, 20, 30, 40], 3);
        assert_eq!(hit_again.matched_indices, vec![100, 101, 102, 103]);

        t.mark_finished_chain(1);
        t.mark_finished_chain(3);
        let mut evicted2 = t.evict(10);
        evicted2.sort_unstable();
        assert_eq!(evicted2, vec![100, 101, 102, 103]);
    }

    #[test]
    fn mark_finished_one_evicts_only_unique_tail() {
        // After one seq finishes, only nodes whose `owners` becomes empty
        // *and* are leaves may admit to the LRU. Internal shared nodes whose
        // owners simply drop from {1,2}→{2} stay off the LRU entirely.
        let mut t = RadixTree::new();
        append_seq(&mut t, 1, &[(1, 10), (2, 11), (3, 12)]);
        // Seq 2 shares [1, 2] then forks with token 9 / idx 20.
        let _ = t.lookup_prefix(&[1, 2], 2);
        t.append_token(2, 9, 20);

        let lru_before = t.lru_len_estimate();
        t.mark_finished_chain(1);
        // Only seq 1's exclusive tail (idx 12) entered the LRU; the shared
        // prefix kept its non-empty owner set and stayed pinned.
        assert_eq!(
            t.lru_len_estimate() - lru_before,
            1,
            "exactly one node (seq 1's exclusive tail) should join LRU"
        );

        let evicted = t.evict(10);
        assert_eq!(evicted, vec![12]);
    }

    #[test]
    fn reused_prefix_after_partial_eviction() {
        // Drive a full eviction of one sequence's tokens, then demonstrate
        // that any surviving shared prefix is still reusable by a fresh seq.
        let mut t = RadixTree::new();
        append_seq(&mut t, 1, &[(7, 70), (8, 71), (9, 72)]);
        // Seq 2 shares [7, 8] then forks.
        let _ = t.lookup_prefix(&[7, 8], 2);
        t.append_token(2, 88, 880);

        // Finish seq 1 and evict aggressively. Seq 2 must keep [7, 8, 88].
        t.mark_finished_chain(1);
        let evicted = t.evict(100);
        assert_eq!(evicted, vec![72], "only seq 1's unique suffix may go");

        // Fresh seq 3 can still reuse [7, 8] from seq 2's pinned chain.
        let hit = t.lookup_prefix(&[7, 8], 3);
        assert_eq!(hit.matched_indices, vec![70, 71]);

        // After all consumers vanish, the surviving prefix becomes reclaimable.
        t.mark_finished_chain(2);
        t.mark_finished_chain(3);
        let mut evicted2 = t.evict(100);
        evicted2.sort_unstable();
        assert_eq!(evicted2, vec![70, 71, 880]);
    }

    #[test]
    fn empty_owners_alone_drives_lru_admission() {
        // The only way a node can be admitted to the LRU is by having
        // `owners.is_empty()` and being a leaf. Walk the full lifecycle
        // to confirm.
        let mut t = RadixTree::new();

        // 1. A single sequence — nothing should be in LRU while it lives.
        append_seq(&mut t, 42, &[(5, 500), (6, 501)]);
        assert_eq!(
            t.lru_len_estimate(),
            0,
            "live owner must keep nodes off LRU"
        );
        // evict() with anything in flight returns nothing.
        assert!(t.evict(10).is_empty());

        // 2. Finish the seq. `owners.is_empty()` becomes true on the
        // leaf, which is the sole driver of LRU admission.
        t.mark_finished_chain(42);
        assert!(
            t.lru_len_estimate() >= 1,
            "empty owners on a leaf must enter LRU"
        );

        // 3. Re-pin via lookup. `add_owner` flipping owners 0→1 must
        // remove the node from the LRU — again, purely by reference count.
        let hit = t.lookup_prefix(&[5, 6], 7);
        assert_eq!(hit.matched_indices, vec![500, 501]);
        let evicted = t.evict(10);
        assert!(
            evicted.is_empty(),
            "non-empty owners must keep node out of LRU; got {:?}",
            evicted
        );

        // 4. Finish the new seq. Eviction must again succeed solely
        // because `owners.is_empty()` is true.
        t.mark_finished_chain(7);
        let mut evicted2 = t.evict(10);
        evicted2.sort_unstable();
        assert_eq!(evicted2, vec![500, 501]);
    }

    #[test]
    fn relookup_resurrects_node_out_of_lru() {
        // Corner case: a node sits in the LRU with stale generation; a new
        // lookup re-pins it before evict pops it. The pop must observe
        // owners non-empty and skip it cleanly without leaking indices.
        let mut t = RadixTree::new();
        append_seq(&mut t, 1, &[(11, 111), (12, 112)]);
        t.mark_finished_chain(1);
        // Pre-eviction: chain is sitting in LRU.
        assert!(t.lru_len_estimate() >= 1);

        // Reuse the prefix — should remove the node from LRU lazily.
        let hit = t.lookup_prefix(&[11, 12], 2);
        assert_eq!(hit.matched_indices, vec![111, 112]);

        // Now evict: the LRU's remembered entry is stale; pop_front_valid
        // skips it via the generation stamp, returning nothing.
        let evicted = t.evict(10);
        assert!(
            evicted.is_empty(),
            "stale LRU entry must not surface; got {:?}",
            evicted
        );
    }

    // ─── lru_total_indices / evict_collect_at_least ──────────────────

    #[test]
    fn lru_total_indices_counts_finished_leaves() {
        let mut t = RadixTree::new();
        // Three independent chains, all finished → 3 leaves * 4 slots each.
        for s in 1..=3u64 {
            for k in 0..4u32 {
                let token = (10 * s as i32) + k as i32;
                let idx = ((s as u32 - 1) * 4) + k;
                t.append_token(s, token, idx);
            }
            t.mark_finished_chain(s);
        }
        assert_eq!(t.lru_total_indices(), 12);
    }

    #[test]
    fn lru_total_indices_counts_finished_long_chain_ancestors() {
        let mut t = RadixTree::new();
        let n = (EDGE_SPLIT_THRESHOLD * 3) as u32;
        for k in 0..n {
            t.append_token(1, k as i32, k);
        }
        t.mark_finished_chain(1);

        assert_eq!(t.lru_total_indices(), n as usize);

        let got = t.evict_collect_at_least(n as usize);
        assert_eq!(got.len(), n as usize);
    }

    #[test]
    fn lru_total_indices_zero_when_all_pinned() {
        let mut t = RadixTree::new();
        append_seq(&mut t, 1, &[(10, 100), (20, 101)]);
        // Active seq → no LRU entries.
        assert_eq!(t.lru_total_indices(), 0);
    }

    #[test]
    fn evict_collect_at_least_returns_at_least_target() {
        let mut t = RadixTree::new();
        for s in 1..=3u64 {
            for k in 0..4u32 {
                let token = (10 * s as i32) + k as i32;
                let idx = ((s as u32 - 1) * 4) + k;
                t.append_token(s, token, idx);
            }
            t.mark_finished_chain(s);
        }
        // Target of 5 must yield at least 5; one chain is 4 → eviction
        // pops the next chain whole, exceeding the target.
        let got = t.evict_collect_at_least(5);
        assert!(got.len() >= 5, "expected ≥ 5 indices, got {}", got.len());
    }

    #[test]
    fn evict_collect_at_least_stops_at_lru_empty() {
        let mut t = RadixTree::new();
        append_seq(&mut t, 1, &[(10, 100), (20, 101)]);
        t.mark_finished_chain(1);
        // Only 2 indices ever existed; asking for 100 must return 2.
        let got = t.evict_collect_at_least(100);
        assert_eq!(got.len(), 2);
        // Subsequent call returns empty.
        let got = t.evict_collect_at_least(1);
        assert!(got.is_empty());
    }
}
