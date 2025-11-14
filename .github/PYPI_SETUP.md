# PyPI Publishing Setup Guide

This guide explains how to set up automatic PyPI publishing for the `openmirlab` organization using GitHub Actions and PyPI's trusted publishing feature.

## Prerequisites

1. You must have admin access to the `openmirlab` GitHub organization
2. You must have maintainer/owner access to the `mt3-infer` PyPI project
3. The PyPI project must already exist (or you need to create it)

## Step 1: Set Up PyPI Trusted Publishing

### For Organization-Level Publishing:

1. **Log in to PyPI** with an account that has maintainer/owner access to the `mt3-infer` project
   - Go to https://pypi.org/manage/account/

2. **Navigate to Trusted Publishing**:
   - Go to: https://pypi.org/manage/account/publishing/
   - Or: Account Settings → Manage API tokens → Trusted publishing

3. **Add a new trusted publisher**:
   - Click "Add a new pending publisher"
   - **PyPI project name**: `mt3-infer`
   - **Owner**: `openmirlab` (your GitHub organization)
   - **Repository name**: `mt3-infer` (or `openmirlab/mt3-infer`)
   - **Workflow filename**: `.github/workflows/publish.yml`
   - **Environment name**: `pypi` (optional, but recommended)

4. **Save the configuration**

## Step 2: Configure GitHub Environment (Optional but Recommended)

1. **Go to your GitHub organization**:
   - Navigate to: https://github.com/organizations/openmirlab/settings/environments

2. **Create a new environment**:
   - Click "New environment"
   - **Name**: `pypi`
   - **Deployment branches**: You can restrict to `main`/`master` branch if desired

3. **Add protection rules** (optional):
   - Required reviewers: Add team members who should approve releases
   - Wait timer: Add a delay before deployment (optional)

4. **Save the environment**

## Step 3: Verify the Workflow

The workflow file (`.github/workflows/publish.yml`) is already configured. It will:

- Trigger automatically when you publish a GitHub Release
- Build the package using `hatchling` (as specified in `pyproject.toml`)
- Publish to PyPI using trusted publishing (no API tokens needed!)

## Step 4: Publishing a New Version

### Method 1: Using GitHub Releases (Recommended)

1. **Update version in `pyproject.toml`**:
   ```toml
   version = "0.2.0"  # Update to your new version
   ```

2. **Commit and push**:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.2.0"
   git push origin main
   ```

3. **Create a GitHub Release**:
   - Go to: https://github.com/openmirlab/mt3-infer/releases/new
   - **Tag**: `v0.2.0` (must match version in pyproject.toml, with 'v' prefix)
   - **Release title**: `v0.2.0` or a descriptive title
   - **Description**: Add release notes
   - Click "Publish release"

4. **Monitor the workflow**:
   - Go to: https://github.com/openmirlab/mt3-infer/actions
   - The "Publish to PyPI" workflow should start automatically
   - Wait for it to complete (usually 2-3 minutes)

5. **Verify on PyPI**:
   - Check: https://pypi.org/project/mt3-infer/
   - Your new version should appear within a few minutes

### Method 2: Manual Trigger

1. Go to: https://github.com/openmirlab/mt3-infer/actions/workflows/publish.yml
2. Click "Run workflow"
3. Select branch (usually `main`)
4. Optionally specify a version
5. Click "Run workflow"

## Troubleshooting

### Workflow fails with "403 Forbidden" or authentication error

- **Check PyPI trusted publishing**: Ensure the trusted publisher is correctly configured on PyPI
- **Verify organization name**: Make sure it's exactly `openmirlab` (case-sensitive)
- **Check repository name**: Should be `mt3-infer` or `openmirlab/mt3-infer`
- **Workflow filename**: Must match `.github/workflows/publish.yml` exactly

### Workflow fails with "Environment not found"

- **Create the environment**: Go to organization settings → Environments → Create `pypi` environment
- **Or remove environment**: Edit `.github/workflows/publish.yml` and remove the `environment:` section

### Package builds but doesn't appear on PyPI

- **Check PyPI project name**: Must match exactly `mt3-infer` (case-sensitive)
- **Verify permissions**: Ensure your PyPI account has maintainer/owner access
- **Check workflow logs**: Look for specific error messages in the Actions tab

### Version mismatch errors

- **Ensure tag matches version**: If version in `pyproject.toml` is `0.2.0`, tag should be `v0.2.0`
- **Check version format**: PyPI requires semantic versioning (e.g., `0.1.0`, `0.2.0`, `1.0.0`)

## Security Notes

- **Trusted publishing is secure**: No API tokens are stored in GitHub secrets
- **OIDC-based**: Uses OpenID Connect for authentication
- **Organization-level**: Publishing is tied to the organization, not individual accounts
- **Audit trail**: All publishing actions are logged on PyPI

## Additional Resources

- [PyPI Trusted Publishing Documentation](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions OIDC](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect)
- [Python Packaging Guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-uploads-using-github-actions-ci-cd-workflows/)
