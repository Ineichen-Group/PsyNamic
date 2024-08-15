show-jobs:
	squeue -u $(USER)

sort-jobs:
	squeue -S t,Q | less

stop-all-jobs:
	scancel -u $(USER)

max-ram:
	@echo -n "Enter the job ID or job name: " && read jobid && bash -c 'if [[ $$jobid =~ ^[0-9]+$$ ]]; then sstat -a $$jobid -o Jobid,MaxRSS,AveCPU; else sstat -a --name=$$jobid -o Jobid,MaxRSS,AveCPU; fi'

stop-job:
	@echo -n "Enter the job ID or job name: " && read jobid && bash -c 'if [[ $$jobid =~ ^[0-9]+$$ ]]; then scancel $$jobid; else scancel --name=$$jobid; fi'

# 10 Minute GPU session
interactive-gpu:
	srun --pty -n 1 -c 2 --time=00:10:00 --gpus=1 --mem=8G bash -l

interactive-cpu:
	srun --pty -n 1 -c 2 --time=00:10:00 --mem=7G bash -l