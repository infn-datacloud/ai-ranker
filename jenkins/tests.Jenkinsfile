#!groovy
@Library('jenkins-libraries') _

pipeline {
    agent { label 'jenkins-node-label-1' }

    environment {
        COVERAGE_DIR = 'coverage-reports'
        SONAR_HOST = 'https://sonarcloud.io'
        SONAR_ORGANIZATION = 'infn-datacloud'
        SONAR_PROJECT = 'ai-ranker-training'
        SONAR_TOKEN = credentials('sonar-token')
    }

    stages {
        stage('Run tests on multiple python versions') {
            parallel {
                stage('Run tests on python3.10') {
                    steps {
                        script {
                            pythonProject.testCode(
                                pythonVersion: '3.10',
                                coveragercId: '.coveragerc',
                                coverageDir: "${COVERAGE_DIR}",
                                imageIsSlim: false
                                )
                        }
                    }
                }
                stage('Run tests on python3.11') {
                    steps {
                        script {
                            pythonProject.testCode(
                                pythonVersion: '3.11',
                                coveragercId: '.coveragerc',
                                coverageDir: "${COVERAGE_DIR}",
                                imageIsSlim: false
                                )
                        }
                    }
                }
            }
        }
    }
    post {
        always {
            script {
                sonar.analysis(
                    sonarToken: '${SONAR_TOKEN}',
                    sonarProject: "${SONAR_PROJECT}",
                    sonarOrganization: "${SONAR_ORGANIZATION}",
                    sonarHost: "${SONAR_HOST}",
                    coverageDir: "${COVERAGE_DIR}",
                    srcDir: 'src',
                    testsDir: 'tests',
                    pythonVersions: '3.10, 3.11'
                )
            }
        }
    }
}
