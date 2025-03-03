#!groovy
@Library('jenkins-libraries') _

pipeline {
    agent { label 'jenkins-node-label-1' }
    stages {
        stage('Run tests') {
            parallel {
                stage('Python 3.10') {
                    steps {
                        script {
                            pythonProject.formatCode(
                                pythonVersion: '3.10',
                                srcDir: 'src',
                                imageIsSlim: false
                                )
                        }
                    }
                }
                stage('Python 3.11') {
                    steps {
                        script {
                            pythonProject.formatCode(
                                pythonVersion: '3.11',
                                srcDir: 'src',
                                imageIsSlim: false
                                )
                        }
                    }
                }
            }
        }
    }
}
