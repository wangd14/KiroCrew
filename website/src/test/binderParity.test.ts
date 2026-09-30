// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest'
import { dashboardDocument } from '../pages/chat/command-center/dashboardDocument'

// jsdom parses CSS rules but lacks the constructed-sheet replaceSync the card path
// uses; without an adapter that path fails closed and strips EVERY <style>, which would
// mask whether the binder's style skip does any work. This mirrors the adapter in
// dashboardDocument.test.ts: it accepts one top-level fixture rule using jsdom's own
// CSSOM parser, so a bound <style> survives to the binding loop the same way it does in
// the browser.
const ParsedSheet = CSSStyleSheet
function installConstructedSheet() {
  vi.stubGlobal('CSSStyleSheet', class extends ParsedSheet {
    replaceSync(css: string) {
      while (this.cssRules.length) this.deleteRule(0)
      if (css.trim()) this.insertRule(css, 0)
    }
  })
}
afterEach(() => { vi.unstubAllGlobals() })

// The Python parity reader (src/kiro_crew/dashboard_templates/parity.py, html_fields)
// re-implements THREE behaviours of the real binder in dashboardDocument.ts so the
// authoring gates can predict which cells a page fills without a browser:
//
//   1. `style` is skipped rather than filled,
//   2. the scope is the body's DESCENDANTS, so the root and the body itself are never
//      matched,
//   3. the field lookup is EXACT and untrimmed, so a padded attribute never matches.
//
// The only thing tying that Python model to the real binder is two literal strings
// grepped out of this file: the selector and the attribute name. A grep proves the
// strings still exist; it proves nothing about behaviour, so the binder could change
// how it assigns text and every Python-side gate would stay green while generated
// pages render blank cells.
//
// This test closes the gap on the side that owns the behaviour. It drives fixture
// pages through the REAL binder and reads back which cells the binder actually filled,
// then compares that observed set against the same answers the Python reader gives for
// the same page. The observation does not read the binder's source: it gives every
// candidate field a UNIQUE sentinel value and asks which sentinels landed as an
// element's text. A field the binder never matched (skipped `style`, the root or body
// itself, or a padded key the exact lookup misses) keeps its original markup text and
// never carries its sentinel, so it is absent from the observed set.

/** The binding attribute the host reads -- one attribute, one field, one value. */
const BINDING = 'data-dashboard-field'

/**
 * The set of fields the real binder fills for `html`, observed rather than read from
 * source. Each candidate field name found in the markup is bound to a sentinel that
 * cannot occur in the fixture text; a field whose sentinel is the text of some element
 * in the rendered card was filled, and is returned. `data` is the automatic-card path,
 * so the binder runs.
 */
function boundFields(html: string): Set<string> {
  // Collect the candidate names the way the Python reader collects them: every value of
  // the binding attribute in the markup, raw (untrimmed) and unique.
  const parsed = new DOMParser().parseFromString(html, 'text/html')
  const candidates = new Set<string>()
  for (const el of parsed.querySelectorAll(`[${BINDING}]`)) {
    candidates.add(el.getAttribute(BINDING) ?? '')
  }
  // A sentinel per candidate that the fixtures never contain as literal text.
  const sentinel = (field: string) => `\u241E-bound-${[...candidates].indexOf(field)}-\u241E`
  const data: Record<string, string> = {}
  for (const field of candidates) data[field] = sentinel(field)

  installConstructedSheet()
  const rendered = new DOMParser().parseFromString(dashboardDocument(html, {}, 'dark', data), 'text/html')
  const shownText = new Set<string>()
  for (const el of rendered.querySelectorAll('*')) shownText.add(el.textContent ?? '')

  const filled = new Set<string>()
  for (const field of candidates) {
    if (shownText.has(sentinel(field))) filled.add(field)
  }
  return filled
}

/**
 * The fields the Python parity reader (`parity.html_fields`) reports for the same page.
 * This mirrors html_fields' three ACCEPT rules against the real binder's reach; a
 * fixture that would make html_fields RAISE (unclosed markup, empty or repeated
 * bindings, a binding nested in a bound element) is out of scope here -- those are
 * refusals, not a field set, and the Python side owns proving them. The equality gate
 * this test protects only ever compares field SETS, so those cases never reach it.
 */
function parityHtmlFields(html: string): Set<string> {
  const parsed = new DOMParser().parseFromString(html, 'text/html')
  const fields = new Set<string>()
  // Rule 2: the binder asks the BODY for its descendants, so only elements inside the
  // body are reachable; the root and the body itself are not their own descendants.
  for (const el of parsed.body.querySelectorAll(`[${BINDING}]`)) {
    // Rule 1: `style` is skipped before the field is read.
    if (el.tagName.toLowerCase() === 'style') continue
    const raw = el.getAttribute(BINDING) ?? ''
    // Rule 3: the lookup is exact and untrimmed, so a padded name never matches its key.
    if (raw.trim() !== raw || raw === '') continue
    fields.add(raw)
  }
  return fields
}

/** Assert the observed binder behaviour equals the Python reader's answer for a page. */
function expectParity(html: string): Set<string> {
  const observed = boundFields(html)
  const modelled = parityHtmlFields(html)
  expect([...observed].sort()).toEqual([...modelled].sort())
  return observed
}

describe('dashboard binder parity: the real binder matches the Python reader model', () => {
  it('fills a plain body descendant, the ordinary case both sides agree on', () => {
    const fields = expectParity(`<p ${BINDING}="result">old</p>`)
    expect(fields).toEqual(new Set(['result']))
  })

  it('skips a style element rather than filling it (behaviour 1)', () => {
    // A binding on <style> is reachable by the selector but the binder skips it, so its
    // field is never filled. The reader must agree, or a page authoring it ships a hole.
    // The <style> follows visible content so the parser keeps it INSIDE the body; a
    // <style> with nothing before it is hoisted into <head> and would be unreachable by
    // the body scope alone, which would hide whether the skip itself does any work.
    const html = `<p ${BINDING}="result">old</p><style ${BINDING}="css">.x{}</style>`
    const fields = expectParity(html)
    expect(fields).toEqual(new Set(['result']))
    expect(fields.has('css')).toBe(false)
  })

  it('never matches the body or the root themselves (behaviour 2)', () => {
    // The parser reparents a binding on <html>/<body> onto those elements. The binder
    // queries body.querySelectorAll, which excludes the body and the root, so neither is
    // filled even though the attribute is present on them.
    const html = `<html ${BINDING}="root"><body ${BINDING}="body"><p ${BINDING}="inner">old</p></body></html>`
    const fields = expectParity(html)
    expect(fields).toEqual(new Set(['inner']))
    expect(fields.has('root')).toBe(false)
    expect(fields.has('body')).toBe(false)
  })

  it('misses a padded attribute because the lookup is exact and untrimmed (behaviour 3)', () => {
    // The binder looks the attribute value up in the data with no trim. The sanitizer
    // trims the attribute to "lede" before the binder sees it, and the data is keyed by
    // the name the author DECLARED (" lede "), so the lookup misses and the cell renders
    // blank -- exactly what the reader predicts by excluding a padded binding from its
    // set. boundFields keys its sentinels by the declared name, so it observes the same
    // blank cell the reader models.
    const html = `<p ${BINDING}=" lede ">old</p><p ${BINDING}="tidy">old</p>`
    const fields = expectParity(html)
    expect(fields).toEqual(new Set(['tidy']))
    expect(fields.has(' lede ')).toBe(false)
    expect(fields.has('lede')).toBe(false)
  })

  it('agrees on a page that combines all three behaviours at once', () => {
    const html =
      `<html ${BINDING}="root">` +
      `<body ${BINDING}="body">` +
      `<h1 ${BINDING}="title">old</h1>` +
      `<style ${BINDING}="css">.x{}</style>` +
      `<p ${BINDING}=" padded ">old</p>` +
      `<section><span ${BINDING}="nested">old</span></section>` +
      `</body></html>`
    const fields = expectParity(html)
    expect(fields).toEqual(new Set(['title', 'nested']))
  })

  it('fills every reachable field and only those, so the two surfaces cannot silently drift', () => {
    // A regression that widened the binder's reach (e.g. querying the whole document, or
    // stopping the style skip, or trimming the key) would fill a cell this model does not
    // predict; a regression that narrowed it would leave a predicted cell blank. Either
    // way boundFields and parityHtmlFields diverge and this assertion fails -- which is
    // the drift the grep-only tie could never catch.
    const html =
      `<p ${BINDING}="a">old</p>` +
      `<div ${BINDING}="b"><em>old</em></div>` +
      `<style ${BINDING}="skip">.x{}</style>`
    expect(boundFields(html)).toEqual(parityHtmlFields(html))
    expect(parityHtmlFields(html)).toEqual(new Set(['a', 'b']))
  })
})
