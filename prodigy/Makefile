build:
	docker compose build

start: build
	docker compose up -d

stop:
	docker compose down

db-export: start
	docker compose exec psych prodigy db-out pubmed_psych > export/$(shell date +%Y%m%d%H%M%S).jsonl

logs:
	docker compose logs --follow psych