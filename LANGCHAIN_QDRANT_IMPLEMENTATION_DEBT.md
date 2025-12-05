# LangChain + Qdrant Implementation Debt

## Transformers cache environment deprecation

The current LangChain + Qdrant stack still exports `TRANSFORMERS_CACHE`, but Transformers >=4.56 already warns that the variable will be removed in v5 in favor of `HF_HOME`. We already rely on `HF_HOME` in most places, so this debt item keeps track of the cut-over work.

- [x] Update developer shells and local services to rely solely on `HF_HOME` / `SENTENCE_TRANSFORMERS_HOME` (`~/.zprofile`, `personal-embeddings-service/Dockerfile`).
- [x] Remove `TRANSFORMERS_CACHE` references from the personal-embeddings-service runtime (`docker-compose.yml`) so the containers mirror the HF_HOME cache path.
- [ ] Remove the legacy `TRANSFORMERS_CACHE` exports that still live in `ragrun_old/` before upgrading Transformers to v5.

Once the final bullet is complete we can bump Transformers without seeing the warning again.
