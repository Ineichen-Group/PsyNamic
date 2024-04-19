build:
	docker compose build
	docker compose up -d

start:
	docker compose up -d

start-50: start
	docker compose exec -d psych sh -c "python -m prodigy span-and-textcat pubmed_psych en ./input/psychedelic_study_50_20240418.jsonl -F ./recipe.py -pt /patterns.jsonl"

start-100: start
	docker compose exec -d psych sh -c "python -m prodigy span-and-textcat pubmed_psych en ./input/psychedelic_study_100_20240418.jsonl -F ./recipe.py -pt /patterns.jsonl"

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