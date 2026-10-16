pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                    python3 -m venv .venv
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                    . .venv/bin/activate
                    python train_model.py
                    test -s iris_model.pkl
                '''
            }
        }

        stage('Retrain Model') {
            steps {
                sh '''
                    . .venv/bin/activate
                    python retrain.py
                    test -s iris_model.pkl
                '''
            }
        }
    }

    post {
        success {
            archiveArtifacts artifacts: 'iris_model.pkl',
                             fingerprint: true
            echo 'MLOps pipeline completed successfully'
        }

        failure {
            echo 'Pipeline failed. Check Console Output.'
        }
    }
}
