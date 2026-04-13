## v0.2.1 (2025-10-01)

### Added
- Prepared API endpoints for ragprep integration
- Documentation: external CHUNKER_PLAN.md (design overview)

### Changed
- Refactors to align server with upcoming chunker pipeline

### Removed
- Large data artifacts taken out of repository

## Unreleased

### Added
- Config flag `ENABLE_LEGACY_ROUTERS` (default: false). When disabled, legacy routers `/assistants`, `/threads`, and `/threads/{thread_id}/messages` are not mounted. (Flag will be removed in future as legacy is purged.)

### Changed
- API composition gated by `ENABLE_LEGACY_ROUTERS`; `/auth`, `/health`, and `/rag/**` remain active.

### Removed
- Purged legacy endpoints and code: `/assistants/**`, `/threads/**`, and `/threads/{thread_id}/messages/**` removed from codebase and router wiring.
