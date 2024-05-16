build:
	docker compose build
	docker compose up -d

start:
	docker compose up -d

start-50:
	docker compose down
	docker compose up -d
	docker compose exec -d psych sh -c "python -m prodigy span-and-textcat pubmed_psych en ./input/psychedelic_study_50_20240418.jsonl -F ./recipe.py -pt ./patterns.txt"

start-100:
	docker compose down
	docker compose up -d
	docker compose exec -d psych sh -c "python -m prodigy span-and-textcat pubmed_psych en ./input/psychedelic_study_100_20240418.jsonl -F ./recipe.py -pt ./patterns.txt"

start-ben:
	docker compose down
	docker compose up -d
	docker compose exec -d psych sh -c "python -m prodigy span-and-textcat pubmed_psych en ./input/psychedelic_study_250_20240425_152801.jsonl -F ./recipe.py -pt ./patterns.txt"

start-julia: 
	docker compose down
	docker compose up -d
	docker compose exec -d psych sh -c "python -m prodigy span-and-textcat pubmed_psych en ./input/psychedelic_study_250_20240425_134820.jsonl -F ./recipe.py -pt ./patterns.txt"

start-bernard:
	docker compose down
	docker compose up -d
	docker compose exec -d psych sh -c "python -m prodigy span-and-textcat pubmed_psych en ./input/psychedelic_study_250_20240425_134821.jsonl -F ./recipe.py -pt ./patterns.txt"

start-pia:
	docker compose down
	docker compose up -d
	docker compose exec -d psych sh -c "python -m prodigy span-and-textcat pubmed_psych en ./input/psychedelic_study_250_20240425_134822.jsonl -F ./recipe.py -pt ./patterns.txt"

stop:
	docker compose down

# db-export: start
# 		docker compose exec psych sh -c "prodigy db-out pubmed_psych > /export/prodigy_export_$$(date +%Y%m%d_%H%M%S).jsonl"

db-export: start
	docker compose exec psych sh -c "prodigy db-out review_all_text > /export/prodigy_export_review_all_text_$(shell date +%Y%m%d_%H%M%S).jsonl"
	docker compose exec psych sh -c "prodigy db-out review_all_token > /export/prodigy_export_review_all_token_$(shell date +%Y%m%d_%H%M%S).jsonl"

db-reset: stop
	rm -r ./prodigy_data/*

logs:
	docker compose logs --follow psych

shell:
	docker compose exec psych bash

review-text: 
	docker compose down
	docker compose up -d
	docker compose exec psych sh -c "prodigy db-in ben_review ./input/prodigy_export_ben_50_20240418_20240501_181325_pattern_removed.jsonl"
	docker compose exec psych sh -c "prodigy db-in pia_review ./input/prodigy_export_pia_50_20240418_20240509_110412_pattern_removed.jsonl"
	docker compose exec psych sh -c "prodigy db-in bernard_review ./input/prodigy_export_bernard_50_20240418_20240516_091455_pattern_removed.jsonl"
	docker compose exec psych sh -c "prodigy db-in julia_review ./input/prodigy_export_julia_50_20240418_20240516_133214_pattern_removed.jsonl"
	docker compose exec -d psych sh -c "prodigy review review_all_text ben_review,pia_review,bernard_review,julia_review --view-id choice"

review-token:
	docker compose down
	docker compose up -d
	docker compose exec psych sh -c "prodigy db-in ben_review ./input/prodigy_export_ben_50_20240418_20240501_181325_pattern_removed.jsonl"
	docker compose exec psych sh -c "prodigy db-in pia_review ./input/prodigy_export_pia_50_20240418_20240509_110412_pattern_removed.jsonl"
	docker compose exec psych sh -c "prodigy db-in bernard_review ./input/prodigy_export_bernard_50_20240418_20240516_091455_pattern_removed.jsonl"
	docker compose exec psych sh -c "prodigy db-in julia_review ./input/prodigy_export_julia_50_20240418_20240516_133214_pattern_removed.jsonl"
	docker compose exec -d psych sh -c "prodigy review review_all_token ben_review,pia_review,bernard_review,julia_review --view-id ner_manual --label \"Application area\",\"Dosage\""

