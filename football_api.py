import requests

def get_live_scores(team_name):
    """
    Fetches live scores for a specified team.

    Parameters:
    - team_name (str): The name of the team for which to fetch live scores.

    Returns:
    - str: A message containing the live score or a message indicating no live matches.
    """
    api_url = "https://api.football-data.org/v2/competitions/PL/matches?status=LIVE"
    headers = {"X-Auth-Token": "your_api_token"}  # Replace 'your_api_token' with your actual API token
    response = requests.get(api_url, headers=headers)
    
    if response.status_code == 200:
        matches = response.json().get('matches', [])
        for match in matches:
            if team_name.lower() in (match['homeTeam']['name'].lower(), match['awayTeam']['name'].lower()):
                return f"{match['homeTeam']['name']} {match['score']['fullTime']['homeTeam']} - {match['score']['fullTime']['awayTeam']} {match['awayTeam']['name']}"
    return "There are no live matches for this team at the moment."
