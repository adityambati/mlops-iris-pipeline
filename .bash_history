git checkout main
git add .
git commit -m "Dump"
git checkout main
clear
git branch -d dev
git branch -D dev
git push origin --delete dev
git checkout -b dev
clear
vim .github/workflows/ci.yml
git checkout main
vim .github/workflows/ci.yml
git pull origin main
clear
vim .github/workflows/ci.yml
git checkout dev
vim .github/workflows/ci.yml
clear
git branch
clear
git branch
vim .github/workflows/ci.yml
clear
git add .
git commit -m "Added test cases"
git push origin dev
mlflow server     --host 0.0.0.0     --port 8100     --allowed-hosts "*"     --backend-store-uri /home/jupyter/mlruns     --default-artifact-root /home/jupyter/mlruns
clear
mlflow server     --host 0.0.0.0     --port 8100     --allowed-hosts "*"     --backend-store-uri sqlite:///mlflow.db     --default-artifact-root file:/home/jupyter/mlruns
clear
mlflow server     --host 0.0.0.0     --port 8100     --allowed-hosts "*"     --backend-store-uri /home/jupyter/mlruns     --default-artifact-root /home/jupyter/mlruns
git branch
git pull origin
git branch
git pull origin/mlflow-week5
git pull
clear
git branch
git pull mlflow-week5
git pull branch mlflow-week5
git fetch origin
clear
git branch
git branch -a
git checkout -b origin/mlflow-week5
clear
dvc remove local_artifacts/model.joblib.dvc
dvc remove local_artifacts/model.joblib.dvc
bash touch .gitignore
touch .gitignore
nano .gitignore
vim .gitignore
clear
git add train.py inference.py .gitignore dvc.lock
git add train.py inference1.py .gitignore dvc.lock
git add *
git commit -m "feat: Integrate MLflow for experiment tracking and registry"
git push origin mlflow-week5
git push origin
git push --set-upstream origin origin/mlflow-week5
clear
python train.py
ls -l
ls -l mlruns/0
rm -rf mlruns
ls -l mlruns/0
clear
python train.py
ls -l mlruns/0
rm -rf mlruns
clear
python train.py
ls -l mlruns/0
rm -rf mlruns
clear
python train.py
ls -l mlruns/0
ls -l mlruns
pwd
clear
rm -rf mlruns
python train.py
python train.py
rm -rf mlruns
python train.py
python train.py
rm -rf mlruns
clear
python train.py
python train.py
rm -rf mlruns
dvc pull data/eval.csv
python inference.py
python inference1.py
python inference1.py
mlflow server     --host 0.0.0.0     --port 8100     --allowed-hosts "*"     --backend-store-uri /home/jupyter/mlruns     --default-artifact-root /home/jupyter/mlruns
clear
mlflow server     --host 0.0.0.0     --port 8100     --allowed-hosts "*"     --backend-store-uri /home/jupyter/mlruns     --default-artifact-root /home/jupyter/mlruns
clear
mlflow server     --host 0.0.0.0     --port 8100     --allowed-hosts "*"     --backend-store-uri /home/jupyter/mlruns     --default-artifact-root /home/jupyter/mlruns
rm -rf mlruns
python train.py
clear
python train.py
clear
rm -rf mlruns
rm -rf mlruns
clear
rm -rf mlruns
clear
python train.py
python train.py
dvc pull data/eval.csv
python inference1.py
git status
git add *
git -f add *
git add -f *
clear
git status
clear
git branch
git commit -m "Updated training pipeline and inference"
clear
git push origin
git branch
git pull origin
git checkout dev
clear
ls
vim .github/workflows/ci.yml
git add *
git commit -m "Added Docker files"
git push dev
git push origin dev
git pull origin dev
clear
git push origin dev
vim .github/workflows/ci.yml
git add *
git add -f *
git commit -m "Updated ci.yml"
git push origin dev
git add .github/workflows/ci.yml
git commit -m "Add CI workflow configuration"
git push origin dev
vim .github/workflows/ci.yml
git add .github/workflows/ci.yml
git commit -m "Updated ci.yml"
git push origin dev
vim .github/workflows/ci.yml
git add .github/workflows/ci.yml
git commit -m "Updated ci.yml again"
git push origin dev
vim .github/workflows/ci.yml
git add download_model.py .github/workflows/ci.yml
git commit -m "fix: Move model download to separate script to fix CI"
git push origin dev
vim .github/workflows/ci.yml
git add .github/workflows/ci.yml
git commit -m "Fix:ci.yml"
git push origin dev
vim .github/workflows/ci.yml
git add .github/workflows/ci.yml
git commit -m "Fix:ci.yml syntax"
git push origin dev
vim .github/workflows/ci.yml
echo "" > .github/workflows/ci.yml
vim .github/workflows/ci.yml
git add .github/workflows/ci.yml
git commit -m "New ci.yml"
git push origin dev
git add *
git add .github/workflows/ci.yml
git commit -m "Updated train.py"
git push origin dev
vim .github/workflows/ci.yml
git add .github/workflows/ci.yml
git commit -m "temp: Test deploy job on dev branch"
git push origin dev
git push origin dev
vim .github/workflows/ci.yml
git add .github/workflows/ci.yml
git commit -m "temp: Test deploy job on dev branch fix1"
git push origin dev
git add app.py requirements.txt Dockerfile deployment.yml download_model.py
git commit -m "feat: Add all application files for API deployment"
git push origin dev
vim .github/workflows/ci.yml
git add .github/workflows/ci.yml
git commit -m "Minor fix"
git push origin dev
vim .github/workflows/ci.yml
git add .github/workflows/ci.yml
git commit -m "revert: Set deploy job back to main branch"
git push origin dev
mlflow server     --host 0.0.0.0     --port 8100     --allowed-hosts "*"     --backend-store-uri /home/jupyter/mlruns     --default-artifact-root /home/jupyter/mlruns
clear
mlflow server     --host 0.0.0.0     --port 8100     --allowed-hosts "*"     --backend-store-uri /home/jupyter/mlruns     --default-artifact-root gs://21f3000274-week2-artifacts/mlflow-artifacts
clear
mlflow server     --host 0.0.0.0     --port 8100     --allowed-hosts "*"     --backend-store-uri sqlite:///mlflow.db     --default-artifact-root gs://21f3000274-week2-artifacts/mlflow-artifacts
clear
mlflow server     --host 0.0.0.0     --port 8100     --allowed-hosts "*"     --backend-store-uri sqlite:///mlflow.db     --default-artifact-root gs://21f3000274-week2-artifacts/mlflow-artifacts
python app.py
python train.py
python train.py
clear
python train.py
clear
python train.py
rm -rf mlruns
python train.py
clear
rm -rf mlruns
python train.py
curl -X POST http://136.112.143.121:80/predict      -H "Content-Type: application/json"      -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

curl -X POST http://136.112.143.121:80/predict      -H "Content-Type: application/json"      -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
