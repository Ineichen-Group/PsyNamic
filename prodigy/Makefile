.PHONY: build start start-pia stop db-export db-reset logs verbose shell clean

build:
	docker compose build
	docker compose up -d

start:
	docker compose up -d

start-pia: stop
	@echo "Starting prodigy 9.0 with 250 samples"
	@docker compose up -d
	@docker compose exec -d psych sh -c "python -m prodigy span-and-textcat pubmed_psych en ./input/psychedelic_study_250_20240730_095458.jsonl -F ./recipe.py -pt ./patterns.txt"

stop:
	docker compose down

db-export: start
	@docker compose exec psych sh -c "if sqlite3 /root/.prodigy/prodigy.db 'SELECT name FROM dataset WHERE name=\"pubmed_psych\";' | grep -q pubmed_psych; then prodigy db-out pubmed_psych > /export/prodigy_export_iaa_$$(date +%Y%m%d_%H%M%S).jsonl && echo 'Exported database pubmed_psych'; fi"
	@docker compose exec psych sh -c "if sqlite3 /root/.prodigy/prodigy.db 'SELECT name FROM dataset WHERE name=\"review_text\";' | grep -q review_text; then prodigy db-out review_text > /export/prodigy_export_review_text_$$(date +%Y%m%d_%H%M%S).jsonl && echo 'Exported database review_all_tex'; fi"
	@docker compose exec psych sh -c "if sqlite3 /root/.prodigy/prodigy.db 'SELECT name FROM dataset WHERE name=\"review_token\";' | grep -q review_token; then prodigy db-out review_token > /export/prodigy_export_review_token_$$(date +%Y%m%d_%H%M%S).jsonl && echo 'Exported database review_token'; fi"

db-reset: stop
	rm -r ./prodigy_data/*

logs:
	docker compose logs --follow psych

verbose: start
	docker compose exec psych sh -c "PRODIGY_LOGGING=verbose prodigy span-and-textcat pubmed_psych en ./input/psychedelic_study_40_20240523_162946.jsonl -F ./recipe.py -pt ./patterns.txt"

shell: start
	docker compose exec psych bash

clean:
	@docker compose down --rmi all --volumes --remove-orphans
	@docker system prune -a --volumes -f
	@docker volume prune -f
	@docker network prune -f

review-text: 
	docker compose down
	docker compose up -d
	docker compose exec psych sh -c "prodigy db-in reannotated_text ./input/prodigy_export_double_annotated_20240812.jsonl"
	docker compose exec -d psych sh -c "prodigy review review_text reannotated_text --view-id choice"

review-token:
	docker compose down
	docker compose up -d
	docker compose exec psych sh -c "prodigy db-in reannotated_token ./input/prodigy_export_double_annotated_20240812.jsonl"
	docker compose exec -d psych sh -c "prodigy review review_token reannotated_token --view-id ner_manual --label \"Application area\",\"Dosage\""