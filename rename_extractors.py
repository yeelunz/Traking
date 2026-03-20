import os
import re

DIR = r'c:\Users\User\Desktop\code\Traking'

renames = [
    # Registered names (strings) and common text references
    (r'\bmotion_texture_static_v2\b', 'tab_v2_extend'),
    (r'\bmotion_texture_static\b', 'tab_v2'),
    (r'\bmotion_static_v3pro\b', 'tab_v3_pro'),
    (r'\btime_series_v3lite\b', 'tsc_v3_lite'),
    (r'\btime_series_v3pro\b', 'tsc_v3_pro'),
    (r'\bmotion_static_lite\b', 'tab_v3_lite'),
    (r'\btime_series_v2\b', 'tsc_v2_extend'),
    
    # Class names
    (r'\bMotionTextureStaticV2FeatureExtractor\b', 'TabV2ExtendFeatureExtractor'),
    (r'\bMotionTextureStaticFeatureExtractor\b', 'TabV2FeatureExtractor'),
    (r'\bMotionStaticV3ProFeatureExtractor\b', 'TabV3ProFeatureExtractor'),
    (r'\bTimeSeriesV3LiteFeatureExtractor\b', 'TscV3LiteFeatureExtractor'),
    (r'\bTimeSeriesV3ProFeatureExtractor\b', 'TscV3ProFeatureExtractor'),
    (r'\bMotionStaticLiteFeatureExtractor\b', 'TabV3LiteFeatureExtractor'),
    (r'\bTimeSeriesV2FeatureExtractor\b', 'TscV2ExtendFeatureExtractor'),
    (r'\bTimeSeriesFeatureExtractor\b', 'TscV2FeatureExtractor'),

    # Target bare time_series where it refers to the extractor
    (r'"time_series"', '"tsc_v2"'),
    (r"'time_series'", "'tsc_v2'"),
    (r'`time_series`', '`tsc_v2`'),
    (r'(\bfeature_extractor\s*:\s*)time_series\b', r'\1tsc_v2'),
    (r'(\bextractor(_name)?\s*(:|=)\s*)time_series\b', r'\1tsc_v2'),

    # Obsolete names
    (r'"motion_texture"', '"DELETED_motion_texture"'),
    (r"'motion_texture'", "'DELETED_motion_texture'"),
    (r'`motion_texture`', '`DELETED_motion_texture`'),
    (r'"motion_only"', '"DELETED_motion_only"'),
    (r"'motion_only'", "'DELETED_motion_only'"),
    (r'`motion_only`', '`DELETED_motion_only`'),
]

extensions = ('.py', '.md', '.yaml', '.txt', '.json')

for root, dirs, files in os.walk(DIR):
    if '.git' in root or '.gemini' in root or '.hg' in root or '__pycache__' in root:
        continue
    for file in files:
        if file.endswith(extensions) and file != 'rename_extractors.py':
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            new_content = content
            for pattern, replacement in renames:
                new_content = re.sub(pattern, replacement, new_content)
            if new_content != content:
                print(f"Updated {path}")
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
print("Rename complete.")
