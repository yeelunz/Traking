$dir = "c:\Users\User\Desktop\code\Traking"
$extensions = @(".py", ".md", ".yaml", ".txt", ".json", ".yml")

$renames = @(
    @{Pattern = '\bmotion_texture_static_v2\b'; Replacement = 'tab_v2_extend'},
    @{Pattern = '\bmotion_texture_static\b'; Replacement = 'tab_v2'},
    @{Pattern = '\bmotion_static_v3pro\b'; Replacement = 'tab_v3_pro'},
    @{Pattern = '\btime_series_v3lite\b'; Replacement = 'tsc_v3_lite'},
    @{Pattern = '\btime_series_v3pro\b'; Replacement = 'tsc_v3_pro'},
    @{Pattern = '\bmotion_static_lite\b'; Replacement = 'tab_v3_lite'},
    @{Pattern = '\btime_series_v2\b'; Replacement = 'tsc_v2_extend'},
    
    @{Pattern = '"time_series"'; Replacement = '"tsc_v2"'},
    @{Pattern = "'time_series'"; Replacement = "'tsc_v2'"},
    @{Pattern = '`time_series`'; Replacement = '`tsc_v2`'},
    @{Pattern = '(\bfeature_extractor\s*:\s*)time_series\b'; Replacement = '${1}tsc_v2'},
    @{Pattern = '(\bextractor(_name)?\s*(:|=)\s*)time_series\b'; Replacement = '${1}tsc_v2'},

    @{Pattern = '\bMotionTextureStaticV2FeatureExtractor\b'; Replacement = 'TabV2ExtendFeatureExtractor'},
    @{Pattern = '\bMotionTextureStaticFeatureExtractor\b'; Replacement = 'TabV2FeatureExtractor'},
    @{Pattern = '\bMotionStaticV3ProFeatureExtractor\b'; Replacement = 'TabV3ProFeatureExtractor'},
    @{Pattern = '\bTimeSeriesV3LiteFeatureExtractor\b'; Replacement = 'TscV3LiteFeatureExtractor'},
    @{Pattern = '\bTimeSeriesV3ProFeatureExtractor\b'; Replacement = 'TscV3ProFeatureExtractor'},
    @{Pattern = '\bMotionStaticLiteFeatureExtractor\b'; Replacement = 'TabV3LiteFeatureExtractor'},
    @{Pattern = '\bTimeSeriesV2FeatureExtractor\b'; Replacement = 'TscV2ExtendFeatureExtractor'},
    @{Pattern = '\bTimeSeriesFeatureExtractor\b'; Replacement = 'TscV2FeatureExtractor'},

    @{Pattern = '"motion_texture"'; Replacement = '"DELETED_motion_texture"'},
    @{Pattern = "'motion_texture'"; Replacement = "'DELETED_motion_texture'"},
    @{Pattern = '`motion_texture`'; Replacement = '`DELETED_motion_texture`'},
    @{Pattern = '"motion_only"'; Replacement = '"DELETED_motion_only"'},
    @{Pattern = "'motion_only'"; Replacement = "'DELETED_motion_only'"},
    @{Pattern = '`motion_only`'; Replacement = '`DELETED_motion_only`'}
)

Get-ChildItem -Path $dir -Recurse -File | Where-Object { 
    $ext = $_.Extension
    $ext -in $extensions -and $_.FullName -notmatch "\\\.git|\\\.gemini|__pycache__|rename_extractors\.py|rename_script\.ps1"
} | ForEach-Object {
    $path = $_.FullName
    $content = [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8)
    $newContent = $content

    foreach ($rule in $renames) {
        $newContent = [regex]::Replace($newContent, $rule.Pattern, $rule.Replacement)
    }

    if ($newContent -cne $content) {
        Write-Host "Updated $($path)"
        [System.IO.File]::WriteAllText($path, $newContent, [System.Text.Encoding]::UTF8)
    }
}
Write-Host "Done"
