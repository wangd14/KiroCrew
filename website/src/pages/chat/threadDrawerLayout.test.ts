/**
 * The open thread must SPLIT the chat surface at a desktop width, not cover it.
 * The defect pinned here: the drawer was `absolute ... right-0` at every width,
 * so at 1600px the parent's text ran under it and the composer was half hidden.
 *
 * These assertions ARE the reflow proof. Flex decides the width, so there is no
 * number this app computes to read back: the transcript loses exactly the
 * drawer's width when the pane is a shrinkable flex child (`flex-1` + `min-w-0`)
 * and the drawer an unshrinkable sibling with a real width in the same row. Both
 * halves are checked here, because jsdom lays nothing out, so a measured
 * assertion would pass on the broken markup too.
 */
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { THREAD_SPLIT_MIN_W, THREAD_DRAWER_W, threadDrawerWrapperClass } from './threadDrawerLayout'

const CHAT_PAGE = readFileSync(join(__dirname, '../ChatPage.tsx'), 'utf8')
const V = `min-[${THREAD_SPLIT_MIN_W}px]:`

describe('thread drawer layout', () => {
  it('overlays below the breakpoint', () => {
    for (const cls of ['absolute', 'right-0', 'w-full', 'max-w-[520px]', 'z-[46]']) {
      expect(threadDrawerWrapperClass.split(' ')).toContain(cls)
    }
  })

  it('becomes a pane of the row above it, undoing every base rule that would not', () => {
    // A split missing any one still covers the transcript. The literals are checked
    // against the constants: Tailwind would not generate an interpolated variant.
    for (const cls of [`${V}shrink-0`, `${V}h-full`, `${V}w-[${THREAD_DRAWER_W}px]`,
      `${V}relative`, `${V}inset-auto`, `${V}max-w-none`, `${V}z-auto`]) {
      expect(threadDrawerWrapperClass).toContain(cls)
    }
    expect(threadDrawerWrapperClass).not.toContain(`${V}static`)
    expect(threadDrawerWrapperClass).not.toMatch(/min-\[\$\{/)
  })

  it('is the class the chat page gives the drawer', () => {
    const upToDrawer = CHAT_PAGE.slice(0, CHAT_PAGE.indexOf('data-testid="chat-thread-drawer"'))
    const open = upToDrawer.lastIndexOf('key="chat-thread-drawer"')
    expect(open).toBeGreaterThan(-1)
    expect(upToDrawer.slice(open)).toContain('className={threadDrawerWrapperClass}')
  })

  it('leaves the transcript a shrinkable child of that row', () => {
    const pane = CHAT_PAGE.slice(CHAT_PAGE.indexOf('ref={setChatPaneEl}'))
    const cls = pane.slice(0, pane.indexOf('>'))
    expect(cls).toContain('min-w-0')
    expect(cls).toMatch(/flex-1|flex-\[1_1_60%\]/)
  })
})
