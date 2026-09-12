$output = @()

$output += "=============================================="
$output += "SYSTEM HARDWARE SPECIFICATIONS"
$output += "Generated: $(Get-Date)"
$output += "=============================================="

$output += ""
$output += "===== OPERATING SYSTEM ====="
$os = Get-CimInstance Win32_OperatingSystem
$output += "OS                  : $($os.Caption)"
$output += "Version             : $($os.Version)"
$output += "Build               : $($os.BuildNumber)"
$output += "Architecture        : $($os.OSArchitecture)"
$output += "Install Date        : $($os.InstallDate)"
$output += "Time Zone           : $((Get-TimeZone).DisplayName)"
$output += "Time Zone ID        : $((Get-TimeZone).Id)"
$output += "Locale              : $((Get-Culture).Name)"

$output += ""
$output += "===== COMPUTER ====="
$computer = Get-CimInstance Win32_ComputerSystem
$output += "Manufacturer        : $($computer.Manufacturer)"
$output += "Model               : $($computer.Model)"
$output += "Total RAM           : $([math]::Round($computer.TotalPhysicalMemory / 1GB, 2)) GB"

$output += ""
$output += "===== CPU ====="
$cpu = Get-CimInstance Win32_Processor
$output += "CPU                 : $($cpu.Name)"
$output += "Cores               : $($cpu.NumberOfCores)"
$output += "Threads             : $($cpu.NumberOfLogicalProcessors)"
$output += "Max Clock Speed     : $($cpu.MaxClockSpeed) MHz"
$output += "L3 Cache            : $([math]::Round($cpu.L3CacheSize / 1024, 2)) MB"
$output += "Virtualization      : $($cpu.VirtualizationFirmwareEnabled)"

$output += ""
$output += "===== RAM MODULES ====="
$ram = Get-CimInstance Win32_PhysicalMemory
foreach ($module in $ram) {
    $output += "Manufacturer        : $($module.Manufacturer)"
    $output += "Part Number         : $($module.PartNumber)"
    $output += "Capacity            : $([math]::Round($module.Capacity / 1GB, 2)) GB"
    $output += "Speed               : $($module.Speed) MHz"
    $output += "Configured Speed    : $($module.ConfiguredClockSpeed) MHz"
    $output += "Slot                : $($module.DeviceLocator)"
    $output += ""
}

$output += "===== GPU ====="
$gpus = Get-CimInstance Win32_VideoController
foreach ($gpu in $gpus) {
    $output += "GPU                 : $($gpu.Name)"
    $output += "VRAM                : $([math]::Round($gpu.AdapterRAM / 1GB, 2)) GB"
    $output += "Driver Version      : $($gpu.DriverVersion)"
    $output += "Video Resolution    : $($gpu.VideoModeDescription)"
    $output += "Current Resolution  : $($gpu.CurrentHorizontalResolution) x $($gpu.CurrentVerticalResolution)"
    $output += ""
}

$output += "===== NVIDIA GPU (nvidia-smi) ====="
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $nvidia = nvidia-smi --query-gpu=name,memory.total,driver_version,utilization.gpu,memory.used,memory.free --format=csv
    $output += $nvidia
} else {
    $output += "nvidia-smi not available."
}

$output += ""
$output += "===== STORAGE ====="
$disks = Get-PhysicalDisk
foreach ($disk in $disks) {
    $output += "Disk                : $($disk.FriendlyName)"
    $output += "Media Type          : $($disk.MediaType)"
    $output += "Bus Type            : $($disk.BusType)"
    $output += "Capacity            : $([math]::Round($disk.Size / 1GB, 2)) GB"
    $output += ""
}

$output += "===== DISK DRIVE DETAILS ====="
$drives = Get-CimInstance Win32_DiskDrive
foreach ($drive in $drives) {
    $output += "Model               : $($drive.Model)"
    $output += "Interface           : $($drive.InterfaceType)"
    $output += "Capacity            : $([math]::Round($drive.Size / 1GB, 2)) GB"
    $output += "Serial Number       : $($drive.SerialNumber)"
    $output += ""
}

$output += "===== DISPLAY / GPU RESOLUTION ====="
foreach ($gpu in $gpus) {
    $output += "$($gpu.Name): $($gpu.CurrentHorizontalResolution) x $($gpu.CurrentVerticalResolution)"
}

$output += ""
$output += "=============================================="
$output += "END OF REPORT"
$output += "=============================================="

$reportPath = Join-Path $PSScriptRoot "../results/Hardware_Specifications.txt"
$output | Out-File -FilePath $reportPath -Encoding UTF8

Write-Host ""
Write-Host "Report created successfully:"
Write-Host $reportPath