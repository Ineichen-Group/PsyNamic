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
