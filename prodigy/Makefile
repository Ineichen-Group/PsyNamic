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

# start-ben:
# 	docker compose down
# 	docker compose up -d
# 	@echo "Starting prodigy 7.0 with 250 samples for Ben"
# 	@docker compose exec -d psych sh -c "python -m prodigy span-and-textcat pubmed_psych en ./input/psychedelic_study_250_20240523_195806.jsonl -F ./recipe.py -pt ./patterns.txt"

start-40:
	docker compose down
	docker compose up -d
	@echo "Starting prodigy 7.0 with 40 samples for IAA"
	@docker compose exec -d psych sh -c "python -m prodigy span-and-textcat pubmed_psych_iaa en ./input/psychedelic_study_40_20240523_195805.jsonl -F ./recipe.py -pt ./patterns.txt"

stop:
	docker compose down

db-export:
	docker compose up -d
	@docker compose exec psych sh -c "if sqlite3 /root/.prodigy/prodigy.db 'SELECT name FROM dataset WHERE name=\"pubmed_psych\";' | grep -q pubmed_psych; then prodigy db-out pubmed_psych > /export/prodigy_export_ben_$$(date +%Y%m%d_%H%M%S).jsonl && echo 'Exported database pubmed_psych'; fi"
	@docker compose exec psych sh -c "if sqlite3 /root/.prodigy/prodigy.db 'SELECT name FROM dataset WHERE name=\"pubmed_psych_iaa\";' | grep -q pubmed_psych_iaa; then prodigy db-out pubmed_psych_iaa > /export/prodigy_export_iaa_$$(date +%Y%m%d_%H%M%S).jsonl && echo 'Exported database pubmed_psych_iaa'; fi"

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
	docker compose exec psych sh -c "prodigy db-in ben_review ./input/prodigy_export_iaa_ben_40_20240523_20240604_094449_pattern_removed.jsonl"
	docker compose exec psych sh -c "prodigy db-in pia_review ./input/prodigy_export_iaa_pia_40_20240523_20240601_155420_pattern_removed.jsonl"
	docker compose exec -d psych sh -c "prodigy review review_all_text ben_review,pia_review --view-id choice"

review-token:
	docker compose down
	docker compose up -d
	docker compose exec psych sh -c "prodigy db-in ben_review ./input/prodigy_export_iaa_ben_40_20240523_20240604_094449_pattern_removed.jsonl"
	docker compose exec psych sh -c "prodigy db-in pia_review ./input/prodigy_export_iaa_pia_40_20240523_20240601_155420_pattern_removed.jsonl"
	docker compose exec -d psych sh -c "prodigy review review_all_token ben_review,pia_review --view-id ner_manual --label \"Application area\",\"Dosage\""

clean:
	docker compose down --rmi all --volumes --remove-orphans
	docker system prune -a --volumes -f
	docker volume prune -f
	docker network prune -f