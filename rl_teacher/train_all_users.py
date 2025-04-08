import psycopg2
import subprocess
from collections import defaultdict
import argparse
import csv

def get_user_data(experiment_name=""):
    try:
        conn = psycopg2.connect(
            dbname='uiadapt', 
            user='postgres', 
            password='dani', 
            host='127.0.0.1', 
            port='5432'
        )
        cursor = conn.cursor()

        # Get users in the experiment
        cursor.execute("""
            SELECT auth_user.id 
            FROM auth_user
            INNER JOIN human_feedback_api_profile ON auth_user.id = human_feedback_api_profile.user_id
            WHERE auth_user.last_login IS NOT NULL AND human_feedback_api_profile.experiment = %s;
        """, (experiment_name,))
        all_users = {row[0] for row in cursor.fetchall()}

        # Get training completion records
        cursor.execute("""
            SELECT user_id, experiment, domain
            FROM human_feedback_api_trainingcompletion
            WHERE experiment = %s;
        """, (experiment_name,))
        
        completed_training = defaultdict(set)  # {user_id: {domain1, domain2, ...}}
        for user_id, _, domain in cursor.fetchall():
            completed_training[user_id].add(domain)

        cursor.close()
        conn.close()

        return all_users, completed_training
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return set(), {}

def run_script(user_id, domains_to_train, dry_run, experiment_name="test"):
    """ Runs the training script for specified domains. """
    for domain in domains_to_train:
        command = [
            "python", "rl_teacher/teach.py", "-e", "UIAdaptation-v0", "-n", experiment_name, "-p", "human", "-L", "10", 
            "-w", "1", "-tep", "10000", "-d", domain, "-c", "4", "-V", "-u", str(user_id)
        ]
        if dry_run:
            print(f"[DRY RUN] Would execute: {' '.join(command)}")
        else:
            try:
                print(f"Executing command for user {user_id} with domain {domain}...")
                subprocess.Popen(command)
            except Exception as e:
                print(f"Error running command for user {user_id} with domain {domain}: {e}")

def get_incomplete_users(all_users, completed_training, all_domains):
    """ Returns users who did not complete training in all domains. """
    incomplete_users = []
    for user_id in all_users:
        completed_domains = completed_training.get(user_id, set())
        if completed_domains != all_domains:
            incomplete_users.append(user_id)
    return incomplete_users

def get_user_experiment_counts(completed_training):
    """ Returns how many times each (user, experiment) appears in the table. """
    user_experiment_counts = {user_id: len(domains) for user_id, domains in completed_training.items()}
    return user_experiment_counts

def export_to_csv(users, completed_training, filename="training_completion.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Username", "Completed Domains"])
        
        for user_id, username in users.items():
            domains = ", ".join(completed_training.get(user_id, []))
            writer.writerow([username, domains])
    print(f"CSV exported: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all users for an experiment.")
    parser.add_argument("-n", "--experiment_name", required=True, help="Name of the experiment")
    parser.add_argument("--dry_run", action="store_true", help="Only display information, do not train")
    args = parser.parse_args()

    experiment_name = args.experiment_name
    dry_run = args.dry_run

    all_domains = {"trips", "courses"}  # Defined set of domains

    # Fetch user data
    all_users, completed_training = get_user_data(experiment_name)

    # Users who did not complete training in all domains
    incomplete_users = get_incomplete_users(all_users, completed_training, all_domains)
    print("Users who did not complete training:", incomplete_users)

    # How many times a user + experiment appears
    user_experiment_counts = get_user_experiment_counts(completed_training)
    print("User training counts:", user_experiment_counts)

    # Train the agent for the completed domains
    for user_id, domains_completed in completed_training.items():
        run_script(user_id, domains_completed, dry_run, experiment_name=experiment_name)
    export_to_csv(all_users, completed_training)
