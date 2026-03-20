$files = @(
"c:\Users\User\Desktop\code\Traking\tracking\classification\feature_extractors_ext.py",
"c:\Users\User\Desktop\code\Traking\tracking\classification\feature_extractors_v3lite.py",
"c:\Users\User\Desktop\code\Traking\tracking\classification\feature_extractors_v3pro.py",
"c:\Users\User\Desktop\code\Traking\tracking\classification\feature_extractors_v3pro_tsc.py",
"c:\Users\User\Desktop\code\Traking\tracking\classification\engine.py",
"c:\Users\User\Desktop\code\Traking\tracking\classification\classifiers_ext.py",
"c:\Users\User\Desktop\code\Traking\ui.py"
)

foreach ($f in $files) {
    if (Test-Path $f) {
        $c = Get-Content $f -Raw -Encoding UTF8
        $c = $c -replace '\bmotion_texture_static_v2\b','tab_v2_extend'
        $c = $c -replace '\bmotion_texture_static\b','tab_v2'
        $c = $c -replace '\bmotion_static_v3pro\b','tab_v3_pro'
        $c = $c -replace '\btime_series_v3lite\b','tsc_v3_lite'
        $c = $c -replace '\btime_series_v3pro\b','tsc_v3_pro'
        $c = $c -replace '\bmotion_static_lite\b','tab_v3_lite'
        $c = $c -replace '\btime_series_v2\b','tsc_v2_extend'
        $c = $c -replace '"time_series"','"tsc_v2"'
        $c = $c -replace '''time_series''','''tsc_v2'''
        $c = $c -replace '\bMotionTextureStaticV2FeatureExtractor\b','TabV2ExtendFeatureExtractor'
        $c = $c -replace '\bMotionTextureStaticFeatureExtractor\b','TabV2FeatureExtractor'
        $c = $c -replace '\bMotionStaticV3ProFeatureExtractor\b','TabV3ProFeatureExtractor'
        $c = $c -replace '\bTimeSeriesV3LiteFeatureExtractor\b','TscV3LiteFeatureExtractor'
        $c = $c -replace '\bTimeSeriesV3ProFeatureExtractor\b','TscV3ProFeatureExtractor'
        $c = $c -replace '\bMotionStaticLiteFeatureExtractor\b','TabV3LiteFeatureExtractor'
        $c = $c -replace '\bTimeSeriesV2FeatureExtractor\b','TscV2ExtendFeatureExtractor'
        $c = $c -replace '\bTimeSeriesFeatureExtractor\b','TscV2FeatureExtractor'
        Set-Content $f -Value $c -Encoding UTF8
    }
}
