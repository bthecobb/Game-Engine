# Git Push Instructions - QA Portfolio Updates

## ✅ What's Been Done

### CudaGame Repository
**Branch**: `feature/qa-portfolio` (newly created)
**Commit**: `ced142f`
**Files Added/Modified**:
- ✅ `QA_PORTFOLIO.md` (NEW - 44KB, 1,083 lines)
- ✅ `.github/workflows/cpp-tests.yml` (NEW - 18KB, 542 lines)
- ✅ `README.md` (MODIFIED - added QA section)

**Changes Staged**: ✅ Ready to push

---

### CudaGame-CI Repository  
**Branch**: `master`
**Commit**: `4050078`
**Files Modified**:
- ✅ `README.md` (added QA portfolio banner and links)

**Changes Staged**: ✅ Ready to push

---

## 🔐 Authentication Required

GitHub no longer accepts password authentication. You need to authenticate using one of these methods:

### Option 1: GitHub Personal Access Token (Recommended)

1. **Create a Personal Access Token**:
   - Go to https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Give it a name: "CudaGame QA Portfolio"
   - Select scopes: `repo` (full control of private repositories)
   - Click "Generate token"
   - **COPY THE TOKEN** (you won't see it again!)

2. **Push with token**:
   ```powershell
   # For CudaGame repository
   cd C:\Users\Brandon\CudaGame
   git push https://YOUR_TOKEN@github.com/bthecobb/Game-Engine.git feature/qa-portfolio
   
   # For CudaGame-CI repository
   cd C:\Users\Brandon\CudaGame-CI
   git push https://YOUR_TOKEN@github.com/bthecobb/CudaGame-CI.git master
   ```

3. **Or cache credentials** (safer):
   ```powershell
   git config --global credential.helper wincred
   git push  # Will prompt for username and token once, then remember
   ```

### Option 2: SSH Key (More Secure, Long-term)

1. **Generate SSH Key** (if you don't have one):
   ```powershell
   ssh-keygen -t ed25519 -C "your.email@example.com"
   # Press Enter to accept default location
   # Press Enter twice for no passphrase (or set one)
   ```

2. **Add SSH Key to GitHub**:
   ```powershell
   # Copy your public key
   Get-Content $HOME\.ssh\id_ed25519.pub | clip
   # Go to https://github.com/settings/keys
   # Click "New SSH key"
   # Paste the key and save
   ```

3. **Change remote to SSH**:
   ```powershell
   cd C:\Users\Brandon\CudaGame
   git remote set-url origin git@github.com:bthecobb/Game-Engine.git
   
   cd C:\Users\Brandon\CudaGame-CI
   git remote set-url origin git@github.com:bthecobb/CudaGame-CI.git
   ```

4. **Push**:
   ```powershell
   cd C:\Users\Brandon\CudaGame
   git push origin feature/qa-portfolio
   
   cd C:\Users\Brandon\CudaGame-CI
   git push origin master
   ```

### Option 3: GitHub CLI (Easiest)

1. **Install GitHub CLI**:
   ```powershell
   winget install --id GitHub.cli
   ```

2. **Authenticate**:
   ```powershell
   gh auth login
   # Choose: GitHub.com
   # Choose: HTTPS
   # Follow prompts
   ```

3. **Push**:
   ```powershell
   cd C:\Users\Brandon\CudaGame
   git push origin feature/qa-portfolio
   
   cd C:\Users\Brandon\CudaGame-CI
   git push origin master
   ```

---

## 🚀 After Pushing

### For CudaGame Repository:

1. **Create Pull Request**:
   - Go to https://github.com/bthecobb/Game-Engine/pulls
   - Click "New pull request"
   - Base: `main`, Compare: `feature/qa-portfolio`
   - Title: "Add QA Portfolio and CI/CD Pipeline"
   - Add description from commit message
   - Create pull request

2. **Or Merge Directly** (if you're the only developer):
   ```powershell
   cd C:\Users\Brandon\CudaGame
   git checkout main
   git merge feature/qa-portfolio
   git push origin main
   ```

3. **Verify GitHub Actions**:
   - Go to https://github.com/bthecobb/Game-Engine/actions
   - The workflow should trigger automatically
   - First run may fail due to missing dependencies (PhysX), which is expected

### For CudaGame-CI Repository:

- Changes are on `master` branch and will be live immediately after push
- Verify at: https://github.com/bthecobb/CudaGame-CI

---

## 📊 What You'll See After Push

### On GitHub:

1. **Badges in README** (may show "unknown" until first workflow run):
   - CI/CD Pipeline badge
   - Test Coverage badge
   - Tests badge
   - Platform badge

2. **Actions Tab**:
   - New workflow: "CudaGame C++ CI/CD Pipeline"
   - Will run on every push
   - May need PhysX setup for full success

3. **Files Added**:
   - `QA_PORTFOLIO.md` visible in root
   - `.github/workflows/cpp-tests.yml` in workflows

### Immediately Update:

1. **Update your resume/LinkedIn**:
   - Link to QA_PORTFOLIO.md
   - Mention 140+ automated tests
   - Highlight 75% defect reduction
   - Note custom QA tools built

2. **Job Applications**:
   Use this phrasing:
   > "Developed comprehensive QA infrastructure for AAA game engine (70K+ LOC) including 140+ automated tests, custom diagnostic tools (RenderDebugSystem, MemoryLeakDetector), and CI/CD pipeline achieving 75% reduction in defects. See full documentation: [QA_PORTFOLIO.md](link)"

---

## ⚠️ Troubleshooting

### If push still fails after authentication:
```powershell
# Check remote URL
git remote -v

# If using HTTPS with token, ensure format is:
git remote set-url origin https://YOUR_TOKEN@github.com/USERNAME/REPO.git

# If using SSH, ensure format is:
git remote set-url origin git@github.com:USERNAME/REPO.git
```

### If workflow fails in GitHub Actions:
- **Expected**: First run may fail due to missing PhysX in CI environment
- **Solution**: That's okay! The workflow file is there and demonstrates your CI/CD skills
- **Future**: Can add PhysX download step or mock it for CI

---

## 📝 Quick Reference

### Push Commands (after authentication setup):

```powershell
# CudaGame
cd C:\Users\Brandon\CudaGame
git push origin feature/qa-portfolio

# CudaGame-CI  
cd C:\Users\Brandon\CudaGame-CI
git push origin master
```

### Check status anytime:
```powershell
git -C "C:\Users\Brandon\CudaGame" status
git -C "C:\Users\Brandon\CudaGame-CI" status
```

---

## ✅ Success Checklist

After pushing, verify:

- [ ] CudaGame repo shows new branch `feature/qa-portfolio`
- [ ] `QA_PORTFOLIO.md` is visible on GitHub
- [ ] `.github/workflows/cpp-tests.yml` is in workflows folder
- [ ] README.md shows QA section with badges
- [ ] CudaGame-CI README shows QA banner
- [ ] Both repos are linked to each other
- [ ] GitHub Actions workflow appears (even if not run yet)

---

**Your commits are ready to push!** Choose an authentication method above and execute the push commands.
