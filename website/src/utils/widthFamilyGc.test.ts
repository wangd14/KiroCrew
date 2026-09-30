// Feature: chat-virtualizer -- the PER-WIDTH height-cache family is bounded.
//
// Each pane width bucket persists its own `vc_heights_<base>:w<bucket>` blob so
// a table measured at one desktop width is never reused at another (see
// HeightIndex / TranscriptScrollShell). That partition is uncapped by design in
// the WIDTH dimension -- but nothing bounded how MANY width blobs one live slot
// retained, so a slot dragged across many widths grew its family without limit
// toward the ~5 MB localStorage quota (the white-screen `storageGc` exists to
// prevent, one dimension deeper). `widthFamilyGc` adds the missing bound: keep
// the N most-recently-touched widths per slot/host base, evict the rest, never
// touch the current/warm scope, and never cap the measurable width.

import { beforeEach, describe, expect, it, vi } from 'vitest'

import { HeightCache, LS_KEY_PREFIX, TOUCHED_AT_KEY, SCHEMA_VERSION_KEY, HEIGHT_SCHEMA_VERSION } from '../hooks/virtualizer/HeightCache'
import {
  MAX_WIDTH_FAMILIES,
  parseWidthScope,
  pruneWidthFamily,
  boundWidthFamilyFor,
} from './widthFamilyGc'

const keyFor = (scope: string) => `${LS_KEY_PREFIX}${scope}`

/** Persist a width blob at `scope` with a definite `lastTouched` stamp so the
 *  recency ordering under test is deterministic (real writes stamp Date.now()).
 */
function persistAt(scope: string, touchedAt: number, height = 120): void {
  const blob: Record<string, number | string> = {
    [SCHEMA_VERSION_KEY]: HEIGHT_SCHEMA_VERSION,
    [TOUCHED_AT_KEY]: touchedAt,
    'row-a': height,
  }
  localStorage.setItem(keyFor(scope), JSON.stringify(blob))
}

const familyBuckets = (base: string): number[] => {
  const out: number[] = []
  for (let i = 0; i < localStorage.length; i++) {
    const k = localStorage.key(i)
    if (!k || !k.startsWith(LS_KEY_PREFIX)) continue
    const p = parseWidthScope(k.slice(LS_KEY_PREFIX.length))
    if (p && p.base === base) out.push(p.bucket)
  }
  return out.sort((a, b) => a - b)
}

beforeEach(() => {
  localStorage.clear()
})

describe('parseWidthScope', () => {
  it('splits a slot/host width scope into base and bucket', () => {
    expect(parseWidthScope('chat-1-1:tables1:w1216')).toEqual({ base: 'chat-1-1:tables1', bucket: 1216 })
    expect(parseWidthScope('slot:tables1:pane:w704')).toEqual({ base: 'slot:tables1:pane', bucket: 704 })
  })

  it('returns null for a non-width key so it is left alone', () => {
    // A bare per-session key with no `:w<digits>` suffix is not a family member.
    expect(parseWidthScope('chat-1-1')).toBeNull()
    expect(parseWidthScope('chat-1-1:tables1')).toBeNull()
    // A trailing `:w` with no digits is not a bucket.
    expect(parseWidthScope('chat-1-1:tables1:w')).toBeNull()
  })

  it('anchors the bucket to the END, so a base containing :w is not mistaken', () => {
    expect(parseWidthScope('chat:w9:tables1:w1216')).toEqual({ base: 'chat:w9:tables1', bucket: 1216 })
  })
})

describe('width-family recency bound', () => {
  it('measures unbounded growth WITHOUT the policy, then bounds it WITH it', () => {
    const base = 'chat-1-1:tables1'
    // A slot dragged across many desktop widths: one blob per 16px bucket.
    const buckets = Array.from({ length: 40 }, (_, i) => 1024 + i * 16)
    buckets.forEach((w, i) => persistAt(`${base}:w${w}`, 1000 + i))
    // Growth is real and uncapped until we prune.
    expect(familyBuckets(base)).toHaveLength(40)

    // Open the most-recent width; the family is bounded to N, sparing it.
    const current = buckets[buckets.length - 1]
    const removed = boundWidthFamilyFor(`${base}:w${current}`)
    expect(removed).toBe(40 - MAX_WIDTH_FAMILIES)
    expect(familyBuckets(base)).toHaveLength(MAX_WIDTH_FAMILIES)
  })

  it('keeps the MOST-RECENTLY-TOUCHED widths and evicts the least recent', () => {
    const base = 'chat-1-1:tables1'
    // Ascending recency: w1024 oldest ... w1600 newest.
    const buckets = Array.from({ length: MAX_WIDTH_FAMILIES + 3 }, (_, i) => 1024 + i * 16)
    buckets.forEach((w, i) => persistAt(`${base}:w${w}`, 1000 + i))

    // Open a fresh current width so recency alone decides the survivors.
    persistAt(`${base}:w2000`, 5000)
    boundWidthFamilyFor(`${base}:w2000`)

    const survivors = familyBuckets(base)
    expect(survivors).toHaveLength(MAX_WIDTH_FAMILIES)
    // The current width is always kept.
    expect(survivors).toContain(2000)
    // The three oldest are gone; the newest of the original set remain.
    expect(survivors).not.toContain(1024)
    expect(survivors).not.toContain(1040)
    expect(survivors).not.toContain(1056)
    expect(survivors).toContain(buckets[buckets.length - 1])
  })

  it('spares the current scope even when its own blob is the OLDEST (warm return)', () => {
    const base = 'chat-1-1:tables1'
    // The warm width was measured long ago and is the stalest by timestamp...
    persistAt(`${base}:w1216`, 1)
    // ...while many newer widths pile up.
    for (let i = 0; i < MAX_WIDTH_FAMILIES + 5; i++) persistAt(`${base}:w${1024 + i * 16}`, 9000 + i)

    // Returning to the warm width re-opens it: it must survive the prune.
    boundWidthFamilyFor(`${base}:w1216`)
    expect(familyBuckets(base)).toContain(1216)
    expect(familyBuckets(base)).toHaveLength(MAX_WIDTH_FAMILIES)
  })

  it('does nothing while the family is at or under the bound', () => {
    const base = 'chat-1-1:tables1'
    for (let i = 0; i < MAX_WIDTH_FAMILIES; i++) persistAt(`${base}:w${1024 + i * 16}`, 1000 + i)
    expect(boundWidthFamilyFor(`${base}:w1024`)).toBe(0)
    expect(familyBuckets(base)).toHaveLength(MAX_WIDTH_FAMILIES)
  })

  it('bounds each slot/host base INDEPENDENTLY, never crossing bases', () => {
    const a = 'chat-1-1:tables1'
    const b = 'chat-1-1:tables1:pane'
    const other = 'chat-2-2:tables1'
    for (let i = 0; i < MAX_WIDTH_FAMILIES + 4; i++) persistAt(`${a}:w${1024 + i * 16}`, 1000 + i)
    for (let i = 0; i < 3; i++) persistAt(`${b}:w${1024 + i * 16}`, 2000 + i)
    for (let i = 0; i < 3; i++) persistAt(`${other}:w${1024 + i * 16}`, 3000 + i)

    boundWidthFamilyFor(`${a}:w${1024 + (MAX_WIDTH_FAMILIES + 3) * 16}`)

    expect(familyBuckets(a)).toHaveLength(MAX_WIDTH_FAMILIES)
    // The host-scoped and the sibling slot's families are untouched.
    expect(familyBuckets(b)).toHaveLength(3)
    expect(familyBuckets(other)).toHaveLength(3)
  })

  it('leaves a non-width vc_heights_ key strictly alone', () => {
    // No `:w<bucket>` suffix -> not a family member -> never a prune candidate.
    localStorage.setItem(keyFor('chat-1-1'), '{}')
    localStorage.setItem(keyFor('chat-1-1:tables1'), '{}')
    const base = 'chat-1-1:tables1'
    for (let i = 0; i < MAX_WIDTH_FAMILIES + 2; i++) persistAt(`${base}:w${1024 + i * 16}`, 1000 + i)

    boundWidthFamilyFor(`${base}:w${1024 + (MAX_WIDTH_FAMILIES + 1) * 16}`)

    expect(localStorage.getItem(keyFor('chat-1-1'))).toBe('{}')
    expect(localStorage.getItem(keyFor('chat-1-1:tables1'))).toBe('{}')
  })

  it('treats an unstamped (pre-policy) blob as oldest', () => {
    const base = 'chat-1-1:tables1'
    // One blob with no TOUCHED_AT stamp, N stamped newer ones.
    localStorage.setItem(keyFor(`${base}:w1024`), JSON.stringify({ [SCHEMA_VERSION_KEY]: HEIGHT_SCHEMA_VERSION, 'row-a': 100 }))
    for (let i = 0; i < MAX_WIDTH_FAMILIES; i++) persistAt(`${base}:w${1200 + i * 16}`, 9000 + i)

    boundWidthFamilyFor(`${base}:w${1200}`)

    // The unstamped one sorts oldest and is the first evicted.
    expect(familyBuckets(base)).not.toContain(1024)
    expect(familyBuckets(base)).toHaveLength(MAX_WIDTH_FAMILIES)
  })

  it('pruneWidthFamily is a no-op when localStorage throws', () => {
    const spy = vi.spyOn(Storage.prototype, 'key').mockImplementation(() => { throw new Error('boom') })
    try {
      expect(pruneWidthFamily('chat-1-1:tables1', 1216)).toBe(0)
    } finally {
      spy.mockRestore()
    }
  })
})

describe('HeightCache lastTouched stamp', () => {
  it('stamps a write and never surfaces the stamp as a row height', () => {
    const before = Date.now()
    const c = new HeightCache('chat-1-1:tables1:w1216')
    c.set('row-a', 300)
    c.flush()
    const blob = JSON.parse(localStorage.getItem(keyFor('chat-1-1:tables1:w1216'))!)
    expect(typeof blob[TOUCHED_AT_KEY]).toBe('number')
    expect(blob[TOUCHED_AT_KEY]).toBeGreaterThanOrEqual(before)
    // Read back: the stamp is not a measurement.
    const reopened = new HeightCache('chat-1-1:tables1:w1216')
    expect(reopened.peek('row-a')).toBe(300)
    expect(reopened.peek(TOUCHED_AT_KEY)).toBeUndefined()
    expect(reopened.size()).toBe(1)
  })
})
