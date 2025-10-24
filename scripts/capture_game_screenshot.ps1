# Script to run game executable, capture screenshot, and kill process

param(
    [string]$ExecutablePath = "..\build\Release\EnhancedGame.exe",
    [string]$ScreenshotPath = "..\screenshots",
    [int]$WaitSeconds = 3
)

# Create screenshots directory if it doesn't exist
if (!(Test-Path $ScreenshotPath)) {
    New-Item -ItemType Directory -Path $ScreenshotPath | Out-Null
    Write-Host "Created screenshots directory: $ScreenshotPath"
}

# Function to capture screenshot
function Capture-Screenshot {
    param([string]$FilePath)
    
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    
    # Get all screens
    $screens = [System.Windows.Forms.Screen]::AllScreens
    
    # Calculate total bounds
    $totalWidth = 0
    $totalHeight = 0
    
    foreach ($screen in $screens) {
        $totalWidth = [Math]::Max($totalWidth, $screen.Bounds.Right)
        $totalHeight = [Math]::Max($totalHeight, $screen.Bounds.Bottom)
    }
    
    # Create bitmap
    $bitmap = New-Object System.Drawing.Bitmap($totalWidth, $totalHeight)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    
    # Capture screen
    $graphics.CopyFromScreen(0, 0, 0, 0, $bitmap.Size)
    
    # Save screenshot
    $bitmap.Save($FilePath, [System.Drawing.Imaging.ImageFormat]::Png)
    
    # Cleanup
    $graphics.Dispose()
    $bitmap.Dispose()
    
    Write-Host "Screenshot saved to: $FilePath"
}

# Start the game
Write-Host "Starting game: $ExecutablePath"
$gameProcess = Start-Process -FilePath $ExecutablePath -PassThru

# Wait for game to fully load
Write-Host "Waiting $WaitSeconds seconds for game to load..."
Start-Sleep -Seconds $WaitSeconds

# Check if process is still running
if (!$gameProcess.HasExited) {
    # Generate timestamp for filename
    $timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
    $screenshotFile = Join-Path $ScreenshotPath "game_screenshot_$timestamp.png"
    
    # Capture screenshot
    Write-Host "Capturing screenshot..."
    Capture-Screenshot -FilePath $screenshotFile
    
    # Kill the game process
    Write-Host "Killing game process..."
    Stop-Process -Id $gameProcess.Id -Force
    
    Write-Host "Done! Screenshot saved as: $screenshotFile"
    
    # Return the screenshot path for further use
    return $screenshotFile
} else {
    Write-Host "Game process already exited!"
}
