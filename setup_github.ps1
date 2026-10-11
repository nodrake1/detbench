# setup_github.ps1 —— 配置 SSH 走 443 端口并推送 DetBench 到 GitHub
#
# 用法（在 PowerShell 里）：
#     cd C:\Users\nichenh\Desktop\detbench
#     powershell -ExecutionPolicy Bypass -File .\setup_github.ps1
#
# 做四件事：
#   1. 把公钥复制到剪贴板（方便你贴到 GitHub 网页）
#   2. 写 ~/.ssh/config，让 github.com 走 443 端口
#   3. 把远程地址从 HTTPS 换成 SSH
#   4. 测试认证并推送
#
# 唯一需要你手动做的事：把公钥贴到 https://github.com/settings/keys

$ErrorActionPreference = "Continue"
$user = "nodrake1"
$repo = "detbench"
$sshDir = Join-Path $env:USERPROFILE ".ssh"
$pubKey = Join-Path $sshDir "id_ed25519.pub"
$privKey = Join-Path $sshDir "id_ed25519"
$sshConfig = Join-Path $sshDir "config"

function Step($n, $t) {
    Write-Host ""
    Write-Host ("=" * 68) -ForegroundColor DarkGray
    Write-Host "步骤 $n  $t" -ForegroundColor Cyan
    Write-Host ("=" * 68) -ForegroundColor DarkGray
}
function Ok($m)   { Write-Host "  [OK]   $m" -ForegroundColor Green }
function Warn($m) { Write-Host "  [注意] $m" -ForegroundColor Yellow }
function Bad($m)  { Write-Host "  [失败] $m" -ForegroundColor Red }

# ---------------------------------------------------------------- #
Step 1 "检查 SSH 密钥"

if (-not (Test-Path $privKey)) { Bad "缺少私钥 $privKey"; exit 1 }
if (-not (Test-Path $pubKey))  { Bad "缺少公钥 $pubKey";  exit 1 }
Ok "私钥存在: $privKey"
Ok "公钥存在: $pubKey"

# ---------------------------------------------------------------- #
Step 2 "复制公钥到剪贴板（并同时存一份到桌面）"

$keyText = (Get-Content $pubKey -Raw).Trim()
try {
    Set-Clipboard -Value $keyText
    Ok "公钥已复制到剪贴板"
} catch {
    Warn "Set-Clipboard 不可用，跳过（下面会打印出来让你手动复制）"
}

# 额外存一份到桌面，方便用记事本打开复制
$desktopCopy = Join-Path ([Environment]::GetFolderPath("Desktop")) "PUBLIC_KEY_贴到GitHub.txt"
try {
    Set-Content -Path $desktopCopy -Value $keyText -Encoding ascii
    Ok "公钥已另存为: $desktopCopy"
} catch {
    Warn "无法写入桌面副本"
}

Write-Host ""
Write-Host "  ↓↓↓ 这就是要贴到 GitHub 的公钥 ↓↓↓" -ForegroundColor Yellow
Write-Host "  $keyText"
Write-Host "  ↑↑↑ 复制这一整行 ↑↑↑" -ForegroundColor Yellow

# ---------------------------------------------------------------- #
Step 3 "配置 SSH：github.com 走 443 端口"

if (-not (Test-Path $sshDir)) { New-Item -ItemType Directory -Force $sshDir | Out-Null }

$configText = ""
if (Test-Path $sshConfig) { $configText = Get-Content $sshConfig -Raw }
if ($configText -notmatch "ssh\.github\.com") {
    $block = "`nHost github.com`n    Hostname ssh.github.com`n    Port 443`n    User git`n"
    Add-Content -Path $sshConfig -Value $block -Encoding utf8
    Ok "已写入 $sshConfig"
} else {
    Ok "配置已存在，跳过"
}

# 顺便把 github.com:443 的主机密钥写进 known_hosts，避免交互提问
try {
    $scan = ssh-keyscan -p 443 -t rsa,ecdsa,ed25519 ssh.github.com 2>$null
    if ($scan) {
        $known = Join-Path $sshDir "known_hosts"
        $existing = if (Test-Path $known) { Get-Content $known -Raw } else { "" }
        if ($existing -notmatch "ssh\.github\.com") {
            Add-Content -Path $known -Value $scan -Encoding utf8
            Ok "已写入 host key 到 known_hosts"
        } else {
            Ok "host key 已存在"
        }
    } else {
        Warn "ssh-keyscan 无输出，稍后会用到 accept-new 兜底"
    }
} catch {
    Warn "ssh-keyscan 失败，稍后会用到 accept-new 兜底"
}

# ---------------------------------------------------------------- #
Step 4 "切换远程地址为 SSH"

$repoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoDir
Ok "工作目录: $repoDir"

git remote set-url origin "git@github.com:$user/$repo.git" 2>$null
if ($LASTEXITCODE -ne 0) {
    git remote add origin "git@github.com:$user/$repo.git"
}
git config --global --unset http.proxy 2>$null | Out-Null
git config --global --unset https.proxy 2>$null | Out-Null

Write-Host "  当前远程配置："
git remote -v | ForEach-Object { Write-Host "    $_" }

# ---------------------------------------------------------------- #
Step 5 "测试 SSH 认证"

Write-Host "  正在连接 github.com:443 ..."
$test = ssh -T -p 443 -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 git@ssh.github.com 2>&1
Write-Host "  $test"

if ($test -match "successfully authenticated") {
    Ok "SSH 认证成功"
} else {
    Bad "SSH 认证未通过"
    Write-Host ""
    Write-Host "  两种情况：" -ForegroundColor Yellow
    Write-Host "    a) 提示 'Permission denied (publickey)' -> 你还没把公钥贴到 GitHub"
    Write-Host "       打开 https://github.com/settings/keys 点 New SSH key，粘贴上面的公钥"
    Write-Host "       贴完重新运行本脚本即可"
    Write-Host "    b) 提示连接超时/被重置 -> 443 端口也被挡了，需要用代理（见脚本末尾说明）"
    exit 1
}

# ---------------------------------------------------------------- #
Step 6 "推送到 GitHub"

git push -u origin main
if ($LASTEXITCODE -eq 0) {
    Ok "推送成功！"
    Write-Host ""
    Write-Host "  仓库地址: https://github.com/$user/$repo" -ForegroundColor Green
    Write-Host ""
    Write-Host "  可以打开浏览器确认，或运行以下命令查看状态："
    Write-Host "    git status"
    Write-Host "    git log --oneline"
} else {
    Bad "推送失败（上面有 git 的报错信息）"
    Write-Host ""
    Write-Host "  常见原因与处理：" -ForegroundColor Yellow
    Write-Host "    1. 仓库还没在 GitHub 上创建"
    Write-Host "       打开 https://github.com/new 建名为 $repo 的仓库"
    Write-Host "       注意：三个初始化选项（README/.gitignore/license）都不要勾"
    Write-Host "    2. 网络中断 -> 见脚本末尾的代理配置"
    Write-Host "    3. 远程已有内容 -> 先执行: git pull --rebase origin main"
    exit 1
}
