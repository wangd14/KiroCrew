/**
 * Bounded recency policy for persisted PER-WIDTH height-cache families.
 *
 * The chat virtualizer scopes each `HeightCache` by session PLUS the pane's
 * settled width bucket (see `measurement.ts` / `TranscriptScrollShell.tsx`), so
 * one raw slot persists a FAMILY of blobs under distinct
 * `vc_heights_<base>:w<bucket>` keys -- one per 16px width bucket the pane has
 * ever settled at. That partition is deliberate and MUST stay uncapped in the
 * width dimension: a table's height genuinely differs per desktop width, so a
 * blob measured at one width is worthless at another and reusing it reads to a
 * user as continuous per-row jumping on resize. Warm return to a previously
 * visited width depends on the old blob still being there.
 *
 * What is NOT bounded today is the NUMBER of width blobs a single LIVE slot
 * retains. `utils/storageGc.ts` keys reclamation on the raw slot id
 * (`key.slice(prefix.length).split(':')[0]`), so it keeps every width family of
 * a live slot and only ever removes a whole slot's tier when the slot itself is
 * deleted. A slot dragged across many desktop widths therefore accumulates
 * width blobs without limit, and localStorage has a hard ~5 MB origin quota:
 * enough of them white-screens the app -- the exact failure `storageGc.ts`
 * exists to prevent, one dimension deeper.
 *
 * This module adds the missing bound: within one slot/host base, keep the
 * `MAX_WIDTH_FAMILIES` most-recently-touched width blobs and evict the rest.
 * It operates ONLY on the persisted `vc_heights_<base>:w<bucket>` derived
 * caches -- never on a live `HeightIndex`, a user draft, or configuration -- and
 * it NEVER caps the measurable width (any width may still be measured; the
 * policy just forgets the least-recently-used widths once too many pile up).
 *
 * WHY RECENCY, NOT RESIZE-TIME DELETION
 * =====================================
 * The task is explicit that the outgoing scope must NOT be deleted on every
 * resize: a reader who nudges a split-pane one bucket and back would lose the
 * warm blob they are about to return to. So eviction is driven by RECENCY
 * across the whole family, not by "a resize happened". The policy runs when a
 * scope is (re)opened -- a construction of `HeightCache` -- and always SPARES
 * the scope being opened (`keepBase`/`keepKey`), so the current width and the
 * warm neighbours a user oscillates between survive; only widths untouched
 * longer than the N most recent are dropped.
 *
 * RECENCY SIGNAL
 * ==============
 * Each blob carries a `lastTouched` epoch-ms stamp written by `HeightCache`
 * (see `TOUCHED_AT_KEY`). A blob with no stamp -- one written by a build before
 * this policy shipped -- sorts as oldest, which is correct: it is by definition
 * the stalest provenance and the right first candidate to reclaim.
 */

import {
  LS_KEY_PREFIX,
  TOUCHED_AT_KEY,
  SCHEMA_VERSION_KEY,
} from '../hooks/virtualizer/HeightCache'

/**
 * How many per-width height blobs to retain within ONE slot/host base.
 *
 * A reader realistically oscillates between a handful of layouts (full window,
 * with a side panel open, a split pane, a phone width), so a small window keeps
 * every layout anyone actually returns to warm while still bounding a slot's
 * width family hard. Each blob is at most HeightCache's own bounded size, so the
 * per-slot ceiling is `MAX_WIDTH_FAMILIES * <one blob>` regardless of how many
 * widths the pane was ever dragged through.
 */
export const MAX_WIDTH_FAMILIES = 8

/**
 * Split a height-cache scope into its width-family BASE and the width bucket.
 *
 * A scope is `<base>:w<digits>` where `<base>` is `<slot>:tables1[:host]`.
 * Returns `null` for any key that is not a width-family member (no `:w<digits>`
 * suffix), so a non-width `vc_heights_` key -- none exist in production today,
 * but a hand-written or future one might -- is left strictly alone.
 *
 * The digits must be the WHOLE remainder after `:w`, anchored to the end, so a
 * base that itself happened to contain `:w...` mid-string cannot be mistaken
 * for the bucket.
 */
export function parseWidthScope(scope: string): { base: string; bucket: number } | null {
  const m = /^(.*):w(\d+)$/.exec(scope)
  if (!m) return null
  return { base: m[1], bucket: Number(m[2]) }
}

/** Read a persisted blob's `lastTouched` stamp, or -1 when it has none (a
 *  pre-policy blob) or cannot be read. -1 sorts oldest so unstamped blobs are
 *  reclaimed before any stamped one. */
function readTouchedAt(storage: Storage, storageKey: string): number {
  let raw: string | null
  try {
    raw = storage.getItem(storageKey)
  } catch {
    return -1
  }
  if (raw === null) return -1
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>
    if (!parsed || typeof parsed !== 'object') return -1
    const v = parsed[TOUCHED_AT_KEY]
    return typeof v === 'number' && Number.isFinite(v) ? v : -1
  } catch {
    return -1
  }
}

/**
 * Prune ONE slot/host base's width family to the `MAX_WIDTH_FAMILIES` most-
 * recently-touched blobs, always sparing `keepKey` (the scope being opened).
 *
 * `base` is the value `parseWidthScope` returns -- everything before the
 * trailing `:w<bucket>`. Only keys under the SAME base are considered, so
 * pruning one slot can never reach another slot's, another host's, or the
 * artifacts gallery's blobs.
 *
 * Returns the number of blobs removed. Best-effort: a storage failure on any
 * one removal is swallowed and the rest proceed.
 */
export function pruneWidthFamily(base: string, keepBucket: number): number {
  if (typeof localStorage === 'undefined') return 0
  const storage = localStorage
  const keepScope = `${base}:w${keepBucket}`
  // Collect this base's width-family members. Enumerate by index because we may
  // remove during a later pass; removing while indexing shifts the rest.
  const members: { storageKey: string; scope: string; touchedAt: number }[] = []
  let len: number
  try {
    len = storage.length
  } catch {
    return 0
  }
  for (let i = 0; i < len; i++) {
    let key: string | null
    try {
      key = storage.key(i)
    } catch {
      continue
    }
    if (!key || !key.startsWith(LS_KEY_PREFIX)) continue
    const scope = key.slice(LS_KEY_PREFIX.length)
    const parsed = parseWidthScope(scope)
    if (!parsed || parsed.base !== base) continue
    members.push({ storageKey: key, scope, touchedAt: readTouchedAt(storage, key) })
  }
  if (members.length <= MAX_WIDTH_FAMILIES) return 0
  // Most-recently-touched first. The kept scope is pinned to the front so it
  // survives even if its own stamp is somehow the oldest (a warm return whose
  // blob was written long ago is exactly the case we must not evict).
  members.sort((a, b) => {
    if (a.scope === keepScope) return -1
    if (b.scope === keepScope) return 1
    return b.touchedAt - a.touchedAt
  })
  let removed = 0
  for (const member of members.slice(MAX_WIDTH_FAMILIES)) {
    if (member.scope === keepScope) continue
    try {
      storage.removeItem(member.storageKey)
      removed++
    } catch {
      /* best-effort */
    }
  }
  return removed
}

/**
 * Prune the width family of `scope`'s base, sparing `scope` itself.
 *
 * The single entry point a `HeightCache` calls after opening a scope. A no-op
 * for a scope with no `:w<bucket>` suffix (nothing to bound), and it never
 * touches the scope it was handed -- the just-opened current width.
 */
export function boundWidthFamilyFor(scope: string): number {
  const parsed = parseWidthScope(scope)
  if (!parsed) return 0
  // Guard the reserved key against being confused for a scope: a scope literally
  // spelled like the version/touch slot is not a real width family.
  if (parsed.base === SCHEMA_VERSION_KEY || parsed.base === TOUCHED_AT_KEY) return 0
  return pruneWidthFamily(parsed.base, parsed.bucket)
}
