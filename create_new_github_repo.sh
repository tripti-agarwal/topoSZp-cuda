#!/bin/bash
# Script to help create a new GitHub repository and push current code

set -e

echo "=========================================="
echo "Creating New GitHub Repository"
echo "=========================================="
echo ""
echo "STEP 1: First, let's check the current status..."
echo ""

cd /u/tagarwal1/TopologySZp/SZp

# Show current status
echo "Current git status:"
git status --short | head -20
echo ""

# Ask if user wants to commit current changes
read -p "Do you want to commit all current changes? (y/n): " commit_changes
if [ "$commit_changes" = "y" ] || [ "$commit_changes" = "Y" ]; then
    echo "Staging all changes..."
    git add -A
    echo "Committing changes..."
    git commit -m "Add topology-preserved compression with error bound preservation"
    echo "✓ Changes committed"
else
    echo "Skipping commit. Make sure to commit or stash changes before pushing."
fi

echo ""
echo "=========================================="
echo "STEP 2: Create a new GitHub repository"
echo "=========================================="
echo ""
echo "Please follow these steps:"
echo "1. Go to https://github.com/new"
echo "2. Create a new repository with your desired name"
echo "3. DO NOT initialize it with README, .gitignore, or license"
echo "4. Copy the repository URL (e.g., https://github.com/username/repo-name.git)"
echo ""

read -p "Enter the new GitHub repository URL: " new_repo_url

if [ -z "$new_repo_url" ]; then
    echo "Error: Repository URL is required"
    exit 1
fi

echo ""
echo "=========================================="
echo "STEP 3: Adding new remote and pushing"
echo "=========================================="
echo ""

# Check if a remote named 'new-origin' already exists
if git remote | grep -q "^new-origin$"; then
    echo "Remote 'new-origin' already exists. Removing it..."
    git remote remove new-origin
fi

# Add the new remote
echo "Adding new remote as 'new-origin'..."
git remote add new-origin "$new_repo_url"

echo ""
echo "Current remotes:"
git remote -v

echo ""
read -p "Do you want to push to the new repository now? (y/n): " push_now

if [ "$push_now" = "y" ] || [ "$push_now" = "Y" ]; then
    echo "Pushing to new repository..."
    git push -u new-origin main
    echo ""
    echo "✓ Successfully pushed to new repository!"
    echo ""
    echo "To make this the default remote, you can run:"
    echo "  git remote rename origin old-origin"
    echo "  git remote rename new-origin origin"
else
    echo ""
    echo "To push later, run:"
    echo "  git push -u new-origin main"
fi

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="

