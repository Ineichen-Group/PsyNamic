PORT=8080

stop-port:
	@if [ "$(OS)" = "Windows_NT" ]; then \
		powershell -File stop_service.ps1 -PORT $(PORT); \
	else \
		bash stop_service.sh $(PORT); \
	fi
build: stop-port
	docker compose build
	docker compose up -d

start:
	docker compose up -d

start-40:
	docker compose down
	docker compose up -d
	@echo "Starting prodigy 8.0 with 40 samples for IAA"
	@docker compose exec -d psych sh -c "python -m prodigy span-and-textcat pubmed_psych_iaa en ./input/psychedelic_study_40_20240620_145100.jsonl -F ./recipe.py -pt ./patterns.txt"

stop:
	docker compose down

db-export:
	docker compose up -d
	@docker compose exec psych sh -c "if sqlite3 /root/.prodigy/prodigy.db 'SELECT name FROM dataset WHERE name=\"pubmed_psych_iaa\";' | grep -q pubmed_psych_iaa; then prodigy db-out pubmed_psych_iaa > /export/prodigy_export_iaa_$$(date +%Y%m%d_%H%M%S).jsonl && echo 'Exported database pubmed_psych_iaa'; fi"
	@docker compose exec psych sh -c "if sqlite3 /root/.prodigy/prodigy.db 'SELECT name FROM dataset WHERE name=\"review_all_text\";' | grep -q review_all_text; then prodigy db-out review_all_text > /export/prodigy_export_review_all_text_$$(date +%Y%m%d_%H%M%S).jsonl && echo 'Exported database review_all_tex'; fi"
	@docker compose exec psych sh -c "if sqlite3 /root/.prodigy/prodigy.db 'SELECT name FROM dataset WHERE name=\"review_all_token\";' | grep -q review_all_token; then prodigy db-out review_all_token > /export/prodigy_export_review_all_token_$$(date +%Y%m%d_%H%M%S).jsonl && echo 'Exported database review_all_token'; fi"

db-reset: stop
	rm -r ./prodigy_data/*

logs:
	docker compose logs --follow psych

verbose: start
	docker compose exec psych sh -c "PRODIGY_LOGGING=verbose prodigy span-and-textcat pubmed_psych en ./input/psychedelic_study_40_20240523_162946.jsonl -F ./recipe.py -pt ./patterns.txt"

shell: start
	docker compose exec psych bash

review-text: 
	docker compose down
	docker compose up -d
	docker compose exec psych sh -c "prodigy db-in ben_review_text ./input/prodigy_export_iaa_ben_40_20240523_20240604_094449_reordered_pattern_removed_text.jsonl"
	docker compose exec psych sh -c "prodigy db-in pia_review_text ./input/prodigy_export_iaa_pia_40_20240523_20240601_155420_reordered_pattern_removed_text.jsonl"
	docker compose exec -d psych sh -c "prodigy review review_all_text ben_review_text,pia_review_text --view-id choice"

review-token:
	docker compose down
	docker compose up -d
	docker compose exec psych sh -c "prodigy db-in ben_review_token ./input/prodigy_export_iaa_ben_40_20240523_20240604_094449_reordered_pattern_removed_token.jsonl"
	docker compose exec psych sh -c "prodigy db-in pia_review_token ./input/prodigy_export_iaa_pia_40_20240523_20240601_155420_reordered_pattern_removed_token.jsonl"
	docker compose exec -d psych sh -c "prodigy review review_all_token ben_review_token,pia_review_token --view-id ner_manual --label \"Application area\",\"Dosage\""

clean:
	@docker compose down --rmi all --volumes --remove-orphans
	@docker system prune -a --volumes -f
	@docker volume prune -f
	@docker network prune -f