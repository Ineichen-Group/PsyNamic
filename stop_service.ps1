param (
    [int]$PORT
)

if (-not $PORT) {
    Write-Output "Usage: .\stop_service.ps1 -PORT <PORT>"
    exit 1
}

# Find containers by port
$CONTAINER_IDS = docker ps --filter "publish=$PORT" --format "{{.ID}}"

if ($CONTAINER_IDS) {
    Write-Output "Stopping and removing the existing containers using port $PORT..."
    $CONTAINER_IDS | ForEach-Object {
        docker stop $_
        docker rm $_
    }
} else {
    Write-Output "No containers using port $PORT found."
}
