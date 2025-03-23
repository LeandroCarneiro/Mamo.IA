import requests

def get_sample_type(sample_id):
    # Fetch sample metadata from GEO
    sample_url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={sample_id}&targ=self&form=text&view=full"
    response = requests.get(sample_url)
    if response.status_code != 200:
        return "Failed to retrieve sample data."
    sample_data = response.text

    # Parse metadata fields
    metadata = {}
    for line in sample_data.split('\n'):
        if line.startswith('!'):
            parts = line.split('=', 1)
            if len(parts) >= 2:
                key = parts[0].strip()
                value = parts[1].strip()
                if key in metadata:
                    metadata[key].append(value)
                else:
                    metadata[key] = [value]

    # Check for library strategy directly
    if '!Sample_library_strategy' in metadata:
        return metadata['!Sample_library_strategy'][0]

    # Check characteristics for keywords
    characteristics = metadata.get('!Sample_characteristics_ch1', [])
    for char in characteristics:
        char_lower = char.lower()
        if 'rna-seq' in char_lower:
            return 'RNA-Seq'
        elif 'chip-seq' in char_lower:
            return 'ChIP-Seq'
        elif 'methylation' in char_lower:
            return 'Methylation profiling'
        # Add other experiment types as needed

    # Check platform technology if available
    platform_id = metadata.get('!Sample_platform_id', [None])[0]
    if platform_id:
        platform_url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={platform_id}&targ=self&form=text&view=full"
        platform_response = requests.get(platform_url)
        if platform_response.status_code == 200:
            platform_data = platform_response.text
            for line in platform_data.split('\n'):
                if line.startswith('!Platform_technology'):
                    tech = line.split('=', 1)[1].strip().lower()
                    if 'sequencing' in tech:
                        return 'Sequencing-based'
                    elif 'array' in tech:
                        return 'Microarray'
                    else:
                        return tech.capitalize()

    # Fallback to sample type or default
    return metadata.get('!Sample_type', ['Unknown'])[0]

# Example usage
sample_id = 'GSM1401009'  # Replace with your GEO sample ID
print(f"Sample {sample_id} type: {get_sample_type(sample_id)}")