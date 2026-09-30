import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { api, type MemberRosterRow } from '../../api/client'
import { crewDisplayName } from '../../components/AgentSelector'
import Modal from '../../components/Modal'
import ErrorNotice from '../../components/ErrorNotice'
import { Btn } from '../../components/ui'
import { findReport } from '../../utils/errorReport'

/**
 * "Delete crewmate" — the confirmation the title row's trash button opens.
 *
 * The write is `DELETE /api/agents/{name}`, the same route the crew manager's
 * danger pane uses, and the body says what THAT route does and no more: it
 * removes the crew record (the roster row), drops the crew from its team,
 * removes its crew log (the Work log tab's record), its uploaded picture and
 * the private template copy bound to it; it leaves the DM transcript, the
 * memory store's files and the workspace on disk. See
 * `api_kirocrew_agent_delete` — the dialog's copy is a promise about that
 * handler, so a change there is a change here.
 *
 * On success the roster query is invalidated (it lives under the
 * `['kirocrew-agents']` prefix the crew manager already invalidates, together
 * with the config query), and the page's own landing effect takes over once
 * the re-read lands without the row: the deleted crewmate's thread gives way
 * exactly as it does for a crewmate deleted anywhere else. Nothing here
 * navigates. A failure renders the server's own message inside the dialog and
 * keeps it open, so the user sees why before deciding again.
 */
export default function DeleteCrewmateDialog({
  open,
  member,
  onClose,
  onDeleted,
}: {
  open: boolean
  member: MemberRosterRow
  onClose: () => void
  onDeleted: (member: MemberRosterRow) => void
}) {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [error, setError] = useState<string | null>(null)
  const label = crewDisplayName(member)

  const remove = useMutation({
    mutationFn: async (): Promise<MemberRosterRow> => {
      await api.deleteKirocrewAgent(member.name)
      return member
    },
    onSuccess: async (removed) => {
      setError(null)
      // The same two invalidations the crew manager's delete performs: the
      // registry prefix covers the roster (`MEMBERS_ROSTER_QUERY_KEY`), the
      // open member's projections and briefing; the config query covers every
      // surface that reads the crew list off the config.
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['kirocrew-agents'] }),
        queryClient.invalidateQueries({ queryKey: ['kirocrewConfig'] }),
      ])
      onDeleted(removed)
    },
    onError: (err) => {
      setError(err instanceof Error ? err.message : String(err))
    },
  })

  const busy = remove.isPending

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={t('pages.membersPage.delete_member')}
      maxWidth={440}
      dismissDisabled={busy}
      footer={(
        <>
          <Btn type="button" onClick={onClose} disabled={busy} data-testid="delete-crewmate-cancel">
            {t('pages.membersPage.cancel')}
          </Btn>
          <Btn type="button" danger onClick={() => remove.mutate()} disabled={busy} data-testid="delete-crewmate-confirm">
            {t('pages.membersPage.delete_member_named', { name: label })}
          </Btn>
        </>
      )}
    >
      <div className="flex flex-col gap-3" data-testid="delete-crewmate-body">
        <p className="m-0 text-[13px] leading-relaxed text-text">
          {t('pages.membersPage.delete_member_body', { name: label })}
        </p>
        {/* No hand-off: the dialog is the decision point, and the failure text
            is the server's own reason for refusing it. */}
        <ErrorNotice
          message={error}
          report={findReport(error ?? undefined)}
          title={t('pages.membersPage.delete_member_failed')}
          onDismiss={() => setError(null)}
          testId="delete-crewmate-error"
        />
      </div>
    </Modal>
  )
}
