import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import MarkdownRenderer from '../components/MarkdownRenderer'

const TABLE = '| Name | Value |\n| --- | --- |\n| alpha | 1 |'
const CELL_WIDTH = 120

// jsdom has no layout, so give every header cell a fixed box. With no layout
// the hook renders no grips at all (see the last test).
describe('markdown table column resize', () => {
  const originals: Record<string, PropertyDescriptor | undefined> = {}
  const stub = (prop: 'offsetWidth' | 'offsetHeight' | 'offsetLeft' | 'offsetTop', get: (el: HTMLElement) => number) => {
    originals[prop] = Object.getOwnPropertyDescriptor(HTMLElement.prototype, prop)
    Object.defineProperty(HTMLElement.prototype, prop, { configurable: true, get() { return get(this as HTMLElement) } })
  }
  beforeEach(() => {
    stub('offsetWidth', el => (el.tagName === 'TH' ? CELL_WIDTH : 0))
    stub('offsetHeight', el => (el.tagName === 'THEAD' ? 36 : 0))
    stub('offsetLeft', el => (el.tagName === 'TH' ? (el as HTMLTableCellElement).cellIndex * CELL_WIDTH : 0))
    stub('offsetTop', () => 0)
  })
  afterEach(() => {
    for (const [prop, d] of Object.entries(originals)) {
      if (d) Object.defineProperty(HTMLElement.prototype, prop, d)
      else delete (HTMLElement.prototype as unknown as Record<string, unknown>)[prop]
    }
  })

  it('puts one grip at the right edge of every header cell, inside the scroll wrapper', () => {
    render(<MarkdownRenderer content={TABLE} />)
    const grips = screen.getAllByTestId('table-column-grip')
    expect(grips).toHaveLength(2)
    expect(grips[1].style.left).toBe(`${2 * CELL_WIDTH - 6}px`)
    expect(grips[0].parentElement).toHaveClass('overflow-x-auto')
    expect(screen.getAllByRole('separator')[0]).toHaveAttribute('aria-valuenow', String(CELL_WIDTH))
  })

  it('leaves the table on auto layout until a column is resized', () => {
    const { container } = render(<MarkdownRenderer content={TABLE} />)
    const table = container.querySelector('table')!
    expect(table.style.tableLayout).toBe('')
    expect(table.querySelector('colgroup')).toBeNull()
  })

  it('widens one column from its laid-out width and the table with it, then resets', () => {
    const { container } = render(<MarkdownRenderer content={TABLE} />)
    const table = container.querySelector('table')!
    const [first] = screen.getAllByRole('separator')
    fireEvent.keyDown(first, { key: 'ArrowRight' })
    expect(table.style.tableLayout).toBe('fixed')
    const cols = Array.from(table.querySelectorAll('col')).map(c => c.style.width)
    expect(cols).toEqual([`${CELL_WIDTH + 16}px`, `${CELL_WIDTH}px`])
    expect(table.style.width).toBe(`${2 * CELL_WIDTH + 16}px`)
    expect(first).toHaveAttribute('aria-valuenow', String(CELL_WIDTH + 16))

    // Back to every laid-out width: back to auto layout, no colgroup.
    fireEvent.keyDown(first, { key: 'Enter' })
    expect(table.style.tableLayout).toBe('')
    expect(table.querySelector('colgroup')).toBeNull()
  })

  it('clamps a column to its minimum width', () => {
    const { container } = render(<MarkdownRenderer content={TABLE} />)
    const [first] = screen.getAllByRole('separator')
    for (let i = 0; i < 10; i++) fireEvent.keyDown(first, { key: 'ArrowLeft', shiftKey: true })
    expect(container.querySelector('col')!.style.width).toBe('48px')
  })

  it('never narrows a column auto layout made wider than the drag cap on a widen gesture', () => {
    const WIDE = 800
    Object.defineProperty(HTMLElement.prototype, 'offsetWidth', {
      configurable: true,
      get() {
        const el = this as HTMLElement
        if (el.tagName !== 'TH') return 0
        return (el as HTMLTableCellElement).cellIndex === 0 ? WIDE : CELL_WIDTH
      },
    })
    const { container } = render(<MarkdownRenderer content={TABLE} />)
    const [first] = screen.getAllByRole('separator')
    expect(first).toHaveAttribute('aria-valuemax', String(WIDE))
    fireEvent.keyDown(first, { key: 'ArrowRight' })
    expect(container.querySelector('col')!.style.width).toBe(`${WIDE}px`)
    fireEvent.keyDown(first, { key: 'ArrowLeft' })
    expect(container.querySelector('col')!.style.width).toBe(`${WIDE - 16}px`)
  })

  it('renders no grips without real layout', () => {
    for (const [prop, d] of Object.entries(originals)) {
      if (d) Object.defineProperty(HTMLElement.prototype, prop, d)
    }
    render(<MarkdownRenderer content={TABLE} />)
    expect(screen.queryAllByTestId('table-column-grip')).toHaveLength(0)
  })
})
