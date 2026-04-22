from controllers.settings_controller import get_current_settings
r = get_current_settings('user')
print('personalities count:', len(r['personalities']))
for k, v in r['personalities'].items():
    print(f"  {k:10s}: temp={v['temperature']} tone={v['tone']:12s} depth={v['depth']}")
print('note:', r['personality_note'])
