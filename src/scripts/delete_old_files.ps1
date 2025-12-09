# 删除13:50之前生成的所有文件
$targetTime = Get-Date "2025-12-07 13:50:00"
$resultsPath = "c:\羊驼\pro\analysis\movie_box_office_analysis\results"

# 搜索所有13:50之前的文件
$oldFiles = Get-ChildItem -Path $resultsPath -Recurse -Force | Where-Object {$_.LastWriteTime -lt $targetTime -and !$_.PSIsContainer}

Write-Host "找到 $($oldFiles.Count) 个需要删除的文件："
$oldFiles | ForEach-Object {Write-Host "- $($_.FullName)"}

# 删除这些文件
if ($oldFiles.Count -gt 0) {
    $oldFiles | Remove-Item -Force
    Write-Host "已删除所有13:50之前生成的文件。"
} else {
    Write-Host "没有找到需要删除的文件。"
}