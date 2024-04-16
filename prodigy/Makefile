build:
	docker compose build
	docker compose up -d

start:
	docker compose up -d

stop:
	docker compose down

db-export: start
		docker compose exec psych sh -c "prodigy db-out pubmed_psych > /export/prodigy_export_$(shell date +%Y%m%d_%H%M%S).jsonl"

db-reset:
	rm -r ./prodigy_data/*

logs:
	docker compose logs --follow psych

shell:
	docker compose exec psych bash