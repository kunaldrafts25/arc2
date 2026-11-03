"""Read-only official ARC2 state; never launches jobs or accesses answer files."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = 'arc-prize-2026-arc-agi-2'


def main():
    token_path = ROOT / '.kaggle/access_token'
    os.environ['KAGGLE_API_TOKEN'] = token_path.read_text(encoding='utf-8-sig').strip()
    os.environ['KAGGLE_CONFIG_DIR'] = str(token_path.parent)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    directory = ROOT / 'reports/arc2/current-state'
    directory.mkdir(parents=True, exist_ok=True)
    pages = [dict(name=p.name, content=p.content) for p in api.competition_list_pages(COMPETITION)]
    (directory / 'pages.json').write_text(json.dumps(pages, indent=2), encoding='utf-8')
    quota = api.quota_view()
    gpu = quota.gpu_quota
    limits = api.competition_get_submission_limits(COMPETITION)
    leaders = api.competition_leaderboard_view(COMPETITION, page_size=5)
    competitions = api.competitions_list(search=COMPETITION)
    report = dict(
        checked_at=datetime.now(timezone.utc).isoformat(), competition=COMPETITION,
        gpu_used_seconds=gpu.time_used.total_seconds(),
        gpu_allowed_seconds=gpu.total_time_allowed.total_seconds(),
        gpu_reserved_seconds=gpu.time_reserved.total_seconds(),
        gpu_available_seconds=max(0, (gpu.total_time_allowed-gpu.time_used-gpu.time_reserved).total_seconds()),
        pay_to_scale_enabled=gpu.is_pay_to_scale_enabled,
        quota_refresh_time=str(quota.quota_refresh_time),
        submission_limits=dict(num_today=limits.num_today, num_total=limits.num_total,
                               num_allowed_now=limits.num_allowed_now),
        leaders=[dict(team=l.team_name, score=l.score, date=str(l.submission_date)) for l in leaders],
        competitions=[dict(title=c.title, deadline=str(c.deadline)) for c in competitions.competitions],
        submissions=[dict(ref=s.ref, status=s.status.name, score=s.public_score,
                          description=s.description) for s in api.competition_submissions(COMPETITION)],
        pages=[dict(name=p['name'], characters=len(p['content'] or '')) for p in pages])
    (directory / 'state.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
