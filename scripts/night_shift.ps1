<#
night_shift.ps1 — Codex 밤샘 멀티세션 러너 (감시 + 중단 재개)

사용:
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\night_shift.ps1
    -Date 2026-07-05      실행할 밤 작업 날짜 (기본: 오늘)
    -Register             30분 간격 자기치유 스케줄러 등록 — 러너가 죽거나
                          Codex 사용량 한도로 전 재시도가 소진돼도 다시 살린다
    -Unregister           스케줄러 해제
    -Status               진행 요약만 출력
    -MaxResumes 8         작업당 재개 시도 한도 (러너 1회 기동 기준)
    -CodexArgs "--full-auto"   Windows에서 샌드박스 오류가 나면
                          "--dangerously-bypass-approvals-and-sandbox" 로 교체
    -PollSec 30           감시 폴링 주기(초)

동작:
  1. context\night\<Date>\task*.md 를 작업 목록으로 읽는다 (/night-shift 스킬이 생성)
  2. 작업마다 git worktree(<WorktreeRoot>\<Date>\<task>, 브랜치 night/<Date>/<task>) 생성
  3. codex exec 를 작업별로 병렬 실행, 로그는 context\night\<Date>\_runner\ 에 기록
  4. 완료 판정 = 해당 워크트리에 context\night\<Date>\<task>.DONE 이 생겼는지
  5. DONE 없이 프로세스가 죽으면 백오프(1→2→4→…→30분) 후 재개:
     로그에서 세션 ID를 찾으면 `codex exec resume <id>`, 못 찾으면
     PROGRESS 파일 기반 재개 프리앰블을 붙인 새 세션으로 이어서 실행
#>
param(
    [string]$Date = (Get-Date -Format 'yyyy-MM-dd'),
    [switch]$Register,
    [switch]$Unregister,
    [switch]$Status,
    [int]$MaxResumes = 8,
    [string]$CodexArgs = "--full-auto",
    [string]$WorktreeRoot = "C:\dev\night",
    [int]$PollSec = 30
)

$ErrorActionPreference = "Stop"
$RepoRoot  = Split-Path -Parent $PSScriptRoot
$NightDir  = Join-Path $RepoRoot ("context\night\" + $Date)
$RunnerDir = Join-Path $NightDir "_runner"
$SchedName = "DACON-night-shift"

function Write-Log([string]$msg) {
    $line = ("[{0}] {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $msg)
    Write-Host $line
    if (Test-Path $RunnerDir) {
        try { Add-Content -Path (Join-Path $RunnerDir "runner.log") -Value $line -Encoding UTF8 -ErrorAction Stop } catch {}
    }
}

if ($Unregister) {
    cmd /c "schtasks /Delete /F /TN $SchedName >nul 2>&1"
    Write-Host "스케줄러 해제: $SchedName"
    exit 0
}

if ($Status) {
    if (-not (Test-Path $NightDir)) { Write-Host "작업 없음: $NightDir"; exit 0 }
    foreach ($tf in (Get-ChildItem $NightDir -Filter "task*.md" | Sort-Object Name)) {
        $id = $tf.BaseName
        $wt = Join-Path (Join-Path $WorktreeRoot $Date) $id
        $doneFile = Join-Path $wt ("context\night\{0}\{1}.DONE" -f $Date, $id)
        $flag = "진행/대기"
        if (Test-Path $doneFile) { $flag = "완료" }
        Write-Host ("{0}: {1}  (worktree: {2})" -f $id, $flag, $wt)
        if (Test-Path $RunnerDir) {
            $log = Get-ChildItem $RunnerDir -Filter ($id + ".a*.out.log") -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime | Select-Object -Last 1
            if ($log) {
                Get-Content $log.FullName -Tail 3 -ErrorAction SilentlyContinue |
                    ForEach-Object { Write-Host ("    | " + $_) }
            }
        }
    }
    exit 0
}

# ---- 사전 점검 ----
$codexCmdInfo = Get-Command codex -ErrorAction SilentlyContinue
if (-not $codexCmdInfo) { Write-Host "codex CLI를 찾을 수 없습니다 (PATH 확인)."; exit 1 }
$CodexPath = $codexCmdInfo.Source
if (-not (Test-Path $NightDir)) {
    Write-Host "밤 작업 폴더 없음: $NightDir — 먼저 /night-shift 스킬로 task*.md를 생성하세요."
    exit 0
}
$taskFiles = @(Get-ChildItem $NightDir -Filter "task*.md" | Sort-Object Name)
if ($taskFiles.Count -eq 0) { Write-Host "task*.md 없음: $NightDir"; exit 0 }
New-Item -ItemType Directory -Force $RunnerDir | Out-Null

# ---- 중복 실행 락 (스케줄러가 30분마다 불러도 한 인스턴스만) ----
$lockFile = Join-Path $RunnerDir "runner.lock"
if (Test-Path $lockFile) {
    $oldPid = Get-Content $lockFile -TotalCount 1
    $alive = $null
    if ($oldPid) { $alive = Get-Process -Id ([int]$oldPid) -ErrorAction SilentlyContinue }
    if ($alive) { Write-Host "이미 실행 중(PID $oldPid) — 이 인스턴스는 종료합니다."; exit 0 }
}
Set-Content -Path $lockFile -Value $PID -Encoding Ascii

if ($Register) {
    $tr = ('powershell -NoProfile -ExecutionPolicy Bypass -File "{0}" -Date {1} -CodexArgs "{2}"' -f $PSCommandPath, $Date, $CodexArgs)
    schtasks /Create /F /TN $SchedName /SC MINUTE /MO 30 /TR $tr | Out-Null
    Write-Log ("자기치유 스케줄러 등록: 30분마다 확인 후 필요 시 재기동 ({0})" -f $SchedName)
}

function Ensure-Worktree([string]$id) {
    $wtBase = Join-Path $WorktreeRoot $Date
    New-Item -ItemType Directory -Force $wtBase | Out-Null
    $wt = Join-Path $wtBase $id
    if (Test-Path (Join-Path $wt ".git")) { return $wt }
    $branch = "night/$Date/$id"
    $branchExists = git -C $RepoRoot branch --list $branch
    if ($branchExists) { git -C $RepoRoot worktree add $wt $branch | Out-Null }
    else { git -C $RepoRoot worktree add -b $branch $wt | Out-Null }
    if (-not (Test-Path $wt)) { throw ("worktree 생성 실패: " + $wt) }
    return $wt
}

function Find-SessionId($t) {
    $logs = Get-ChildItem $RunnerDir -Filter ($t.Id + ".a*.out.log") -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending
    foreach ($lf in $logs) {
        $head = Get-Content $lf.FullName -TotalCount 100 -ErrorAction SilentlyContinue
        foreach ($line in $head) {
            if ($line -match '(?i)session\s*id:?\s*([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})') {
                return $Matches[1]
            }
        }
    }
    return $null
}

function Start-CodexAttempt($t) {
    $t.Attempt++
    $outLog = Join-Path $RunnerDir ("{0}.a{1}.out.log" -f $t.Id, $t.Attempt)
    $errLog = Join-Path $RunnerDir ("{0}.a{1}.err.log" -f $t.Id, $t.Attempt)
    $stdinFile = Join-Path $RunnerDir ($t.Id + ".prompt.txt")

    $baseArgs = @()
    if ($CodexArgs.Trim().Length -gt 0) { $baseArgs = @($CodexArgs.Trim() -split '\s+') }

    if (($t.Attempt -gt 1) -and $t.SessionId) {
        # 1순위 재개: 같은 Codex 세션을 그대로 이어서
        $argList = @("exec", "resume", $t.SessionId) + $baseArgs + @("--cd", $t.Worktree)
        $nudge = ("이전 실행이 중단되어 재개한다. context/night/{0}/PROGRESS-{1}.md 와 git log 를 확인해 " -f $Date, $t.Id) +
                 ("이미 끝난 단계는 건너뛰고 남은 작업을 이어서 완료하라. 전부 끝나면 context/night/{0}/{1}.DONE 을 생성하라." -f $Date, $t.Id)
        Set-Content -Path $stdinFile -Value $nudge -Encoding UTF8
        Write-Log ("{0}: 시도 #{1} — 세션 재개 (session {2})" -f $t.Id, $t.Attempt, $t.SessionId)
    }
    elseif ($t.Attempt -gt 1) {
        # 2순위 재개: 세션 ID를 못 찾음 → PROGRESS 파일 기반 새 세션
        $preamble = (@(
            "이전 Codex 세션이 중간에 끊겼다. 아래 '원본 지시'의 작업을 처음부터 다시 하지 말고 이어서 완료하라.",
            ("먼저 git log --oneline -20 과 context/night/{0}/PROGRESS-{1}.md 를 읽어라." -f $Date, $t.Id),
            "이미 끝난 단계는 건너뛰고 PROGRESS의 '다음 재개 지점'부터 진행하라.",
            "", "=== 원본 지시 ===", ""
        ) -join "`r`n")
        Set-Content -Path $stdinFile -Value ($preamble + (Get-Content $t.PromptFile -Raw -Encoding UTF8)) -Encoding UTF8
        $argList = @("exec") + $baseArgs + @("--cd", $t.Worktree)
        Write-Log ("{0}: 시도 #{1} — 새 세션 (PROGRESS 기반 재개)" -f $t.Id, $t.Attempt)
    }
    else {
        Copy-Item $t.PromptFile $stdinFile -Force
        $argList = @("exec") + $baseArgs + @("--cd", $t.Worktree)
        Write-Log ("{0}: 시도 #{1} — 새 세션" -f $t.Id, $t.Attempt)
    }

    # npm 설치 시 codex가 .cmd/.ps1 셔틀일 수 있음 — 리다이렉트를 쓰려면 각 셸 경유 필요
    $exe = $CodexPath
    $pre = @()
    if ($CodexPath -match '\.(cmd|bat)$') { $exe = "cmd.exe"; $pre = @("/c", $CodexPath) }
    elseif ($CodexPath -match '\.ps1$') { $exe = "powershell"; $pre = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $CodexPath) }

    $t.UsedResume = [bool](($t.Attempt -gt 1) -and $t.SessionId)
    $t.LastStart = Get-Date
    $t.Proc = Start-Process -FilePath $exe -ArgumentList ($pre + $argList) -WorkingDirectory $t.Worktree `
        -RedirectStandardInput $stdinFile -RedirectStandardOutput $outLog -RedirectStandardError $errLog `
        -PassThru -WindowStyle Hidden
    # PS 5.1: -PassThru 프로세스의 ExitCode를 나중에 읽으려면 Handle을 먼저 캐시해야 함
    if ($t.Proc) { $null = $t.Proc.Handle }
    $t.State = "running"
}

# ---- 작업 구성 ----
$tasks = @()
foreach ($tf in $taskFiles) {
    $tasks += [pscustomobject]@{
        Id = $tf.BaseName; PromptFile = $tf.FullName; Worktree = $null; Proc = $null
        Attempt = 0; State = "init"; NextRetry = (Get-Date); SessionId = $null
        UsedResume = $false; LastStart = (Get-Date); ResumeBroken = $false
    }
}

try {
    foreach ($t in $tasks) {
        $t.Worktree = Ensure-Worktree $t.Id
        $doneFile = Join-Path $t.Worktree ("context\night\{0}\{1}.DONE" -f $Date, $t.Id)
        if (Test-Path $doneFile) {
            $t.State = "done"
            Write-Log ("{0}: 이미 완료(DONE 존재) — 건너뜀" -f $t.Id)
            continue
        }
        Start-CodexAttempt $t
    }
    Write-Log ("밤샘 러너 시작 — 작업 {0}개, MaxResumes={1}, 폴링 {2}s, CodexArgs='{3}'" -f $tasks.Count, $MaxResumes, $PollSec, $CodexArgs)

    while ($true) {
        $active = 0
        foreach ($t in $tasks) {
            if (($t.State -eq "done") -or ($t.State -eq "failed")) { continue }
            $doneFile = Join-Path $t.Worktree ("context\night\{0}\{1}.DONE" -f $Date, $t.Id)
            $exited = $true
            if ($t.Proc) { $exited = $t.Proc.HasExited }

            if (Test-Path $doneFile) {
                if ($exited) { $t.State = "done"; Write-Log ("{0}: 완료 (DONE 확인, 시도 {1}회)" -f $t.Id, $t.Attempt); continue }
                $active++; continue   # DONE은 썼지만 아직 마무리 동작 중
            }
            if (-not $exited) { $active++; continue }

            if ($t.State -eq "running") {
                # 프로세스가 DONE 없이 종료됨 → 재개 예약
                if (($t.Attempt - 1) -ge $MaxResumes) {
                    $t.State = "failed"
                    Write-Log ("{0}: 재개 한도({1}회) 소진 — 실패로 표기. 로그: {2}" -f $t.Id, $MaxResumes, $RunnerDir)
                    continue
                }
                $delayMin = [math]::Min(30, [math]::Pow(2, [math]::Max(0, $t.Attempt - 1)))
                $t.NextRetry = (Get-Date).AddMinutes($delayMin)
                $t.State = "waiting"
                $exitCode = "?"
                if ($t.Proc) { $exitCode = $t.Proc.ExitCode }
                # 세션 resume 시도가 즉시(60초 내) 죽으면 resume 문법/세션 미존재 문제로 보고
                # PROGRESS 기반 새 세션 폴백으로 전환 (SessionId 폐기)
                $ranSec = ((Get-Date) - $t.LastStart).TotalSeconds
                if ($t.UsedResume -and ($ranSec -lt 60)) {
                    # resume이 즉시 죽음 = resume 문법/세션 미존재 문제 → 이후 이 작업은 세션 재개 경로 봉인
                    $t.SessionId = $null
                    $t.ResumeBroken = $true
                    Write-Log ("{0}: 세션 재개가 {1:n0}초만에 실패 — 이후 PROGRESS 폴백만 사용" -f $t.Id, $ranSec)
                }
                if ((-not $t.SessionId) -and (-not $t.ResumeBroken)) { $t.SessionId = Find-SessionId $t }
                Write-Log ("{0}: DONE 없이 종료(exit={1}) — {2}분 후 재개 예정" -f $t.Id, $exitCode, $delayMin)
                $active++; continue
            }
            if ($t.State -eq "waiting") {
                if ((Get-Date) -ge $t.NextRetry) { Start-CodexAttempt $t }
                $active++; continue
            }
        }
        if ($active -eq 0) { break }
        Start-Sleep -Seconds $PollSec
    }
}
finally {
    Remove-Item $lockFile -Force -ErrorAction SilentlyContinue
}

# ---- 마무리 ----
$summaryLines = @()
foreach ($t in $tasks) { $summaryLines += ("{0}: {1} (시도 {2}회)" -f $t.Id, $t.State, $t.Attempt) }
Set-Content -Path (Join-Path $RunnerDir "summary.txt") -Value ($summaryLines -join "`r`n") -Encoding UTF8
Write-Log ("전체 종료: " + ($summaryLines -join " | "))

$notDone = @($tasks | Where-Object { $_.State -ne "done" })
if ($notDone.Count -eq 0) {
    cmd /c "schtasks /Delete /F /TN $SchedName >nul 2>&1"
    Write-Log "모든 작업 완료 — 스케줄러가 있었다면 해제됨"
    exit 0
}
Write-Log ("미완료 {0}건 — 스케줄러(-Register)가 등록돼 있으면 30분 내 재기동됨" -f $notDone.Count)
exit 1
