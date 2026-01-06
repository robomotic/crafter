import argparse

import numpy as np
try:
  import pygame
except ImportError:
  print('Please install the pygame package to use the GUI.')
  raise

import crafter


def _linux_rss_kb() -> int:
  try:
    with open('/proc/self/status', 'r') as f:
      for line in f:
        if line.startswith('VmRSS:'):
          parts = line.split()
          # Format: VmRSS: <value> kB
          return int(parts[1])
  except Exception:
    pass
  return -1


def main():
  boolean = lambda x: bool(['False', 'True'].index(x))
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=None)
  parser.add_argument('--area', nargs=2, type=int, default=(64, 64))
  parser.add_argument('--view', type=int, nargs=2, default=(9, 9))
  parser.add_argument('--length', type=int, default=None)
  parser.add_argument('--health', type=int, default=9)
  parser.add_argument('--window', type=int, nargs=2, default=(600, 600))
  parser.add_argument('--size', type=int, nargs=2, default=(0, 0))
  parser.add_argument('--record', type=str, default=None)
  parser.add_argument('--fps', type=int, default=5)
  parser.add_argument('--wait', type=boolean, default=False)
  parser.add_argument('--tutorial', type=boolean, default=False)
  parser.add_argument('--stats', type=boolean, default=True)
  parser.add_argument('--death', type=str, default='reset', choices=[
      'continue', 'reset', 'quit'])
  parser.add_argument('--profile', type=boolean, default=False,
      help='Periodically print FPS and memory usage (RSS).')
  parser.add_argument('--profile-frames', type=int, default=60,
      help='How many frames between profile prints when --profile is enabled.')
  args = parser.parse_args()

  keymap = {
      pygame.K_a: 'move_left',
      pygame.K_d: 'move_right',
      pygame.K_w: 'move_up',
      pygame.K_s: 'move_down',
      pygame.K_SPACE: 'do',
      pygame.K_TAB: 'sleep',

      pygame.K_r: 'place_stone',
      pygame.K_t: 'place_table',
      pygame.K_f: 'place_furnace',
      pygame.K_p: 'place_plant',

      pygame.K_1: 'make_wood_pickaxe',
      pygame.K_2: 'make_stone_pickaxe',
      pygame.K_3: 'make_iron_pickaxe',
      pygame.K_4: 'make_wood_sword',
      pygame.K_5: 'make_stone_sword',
      pygame.K_6: 'make_iron_sword',
  }
  print('Actions:')
  for key, action in keymap.items():
    print(f'  {pygame.key.name(key)}: {action}')

  crafter.constants.items['health']['max'] = args.health
  crafter.constants.items['health']['initial'] = args.health

  size = list(args.size)
  size[0] = size[0] or args.window[0]
  size[1] = size[1] or args.window[1]

  env = crafter.Env(
      area=args.area, view=args.view, length=args.length, seed=args.seed,
      tutorial=args.tutorial)
  env = crafter.Recorder(env, args.record)
  obs = env.reset()
  achievements = set()
  duration = 0
  return_ = 0
  was_done = False
  print('Diamonds exist:', env._world.count('diamond'))

  pygame.init()
  
  # Adjust window size for stats panel if enabled
  stats_panel_width = 300 if args.stats else 0
  screen_width = args.window[0] + stats_panel_width
  screen = pygame.display.set_mode((screen_width, args.window[1]))
  pygame.display.set_caption('Crafter - Game')
  
  # Create a reusable surface for the game view to avoid per-frame allocations
  base_size = (size[0], size[1])
  render_surface = pygame.Surface(base_size).convert()
  
  # Setup fonts for stats panel
  stats_font = None
  stats_font_small = None
  recent_achievements = []
  reward_persistence = 30  # Frames to show reward
  reward_display_timer = 0
  last_reward = 0
  last_reward_step = 0
  if args.stats:
    stats_font = pygame.font.Font(None, 24)
    stats_font_small = pygame.font.Font(None, 16)
  
  clock = pygame.time.Clock()
  running = True
  reward = 0
  unlocked = set()
  frames_since_profile = 0
  while running:

    # Rendering game view.
    image = env.render(base_size)
    # Update the reusable surface with the new frame without creating a new surface
    pygame.surfarray.blit_array(render_surface, image.transpose((1, 0, 2)))
    # Scale only if the window size differs from the base render size
    if base_size != tuple(args.window):
      view_surface = pygame.transform.scale(render_surface, args.window)
      screen.blit(view_surface, (0, 0))
    else:
      screen.blit(render_surface, (0, 0))
    
    # Rendering stats panel on the right side.
    if args.stats:
      # Draw stats panel background
      stats_rect = pygame.Rect(args.window[0], 0, stats_panel_width, args.window[1])
      pygame.draw.rect(screen, (20, 20, 20), stats_rect)
      pygame.draw.line(screen, (100, 100, 100), (args.window[0], 0), (args.window[0], args.window[1]), 2)
      
      y_offset = 15
      x_offset = args.window[0] + 15
      
      # Episode counter
      episode_text = stats_font.render(f'Episode: {env._episode}', True, (200, 200, 255))
      screen.blit(episode_text, (x_offset, y_offset))
      y_offset += 35
      
      # Step counter
      step_text = stats_font.render(f'Step: {env._step}', True, (255, 255, 255))
      screen.blit(step_text, (x_offset, y_offset))
      y_offset += 35
      
      # Health
      health_color = (100, 255, 100) if env._player.health > 5 else (255, 100, 100)
      health_text = stats_font.render(f'Health: {env._player.health}', True, health_color)
      screen.blit(health_text, (x_offset, y_offset))
      y_offset += 35
      
      # Persistent reward feedback
      if reward_display_timer > 0:
        reward_color = (0, 255, 100) if last_reward > 0 else (255, 100, 100)
        reward_text = stats_font.render(f'Reward: +{last_reward:.1f}', True, reward_color)
        step_text = stats_font_small.render(f'@ Step {last_reward_step}', True, (150, 150, 150))
        screen.blit(reward_text, (x_offset, y_offset))
        screen.blit(step_text, (x_offset + 10, y_offset + 25))
        reward_display_timer -= 1
      y_offset += 60
      
      # Achievements
      total_achievements = len(env._player.achievements)
      ach_text = stats_font.render(f'Achievements:\n{len(achievements)}/{total_achievements}', True, (255, 255, 100))
      screen.blit(ach_text, (x_offset, y_offset))
      y_offset += 65
      
      # Recently unlocked achievements
      if unlocked:
        recent_achievements = list(unlocked) + recent_achievements
        recent_achievements = recent_achievements[:3]  # Keep last 3
      
      if recent_achievements:
        recent_label = stats_font_small.render('Recent:', True, (200, 200, 200))
        screen.blit(recent_label, (x_offset, y_offset))
        y_offset += 22
        for ach in recent_achievements:
          ach_display = stats_font_small.render(f'✓ {ach}', True, (100, 255, 100))
          screen.blit(ach_display, (x_offset + 5, y_offset))
          y_offset += 20
    
    pygame.display.flip()
    clock.tick(args.fps)
    frames_since_profile += 1

    # Keyboard input.
    action = None
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        running = False
      elif event.type == pygame.KEYDOWN and event.key in keymap.keys():
        action = keymap[event.key]
    if action is None:
      pressed = pygame.key.get_pressed()
      for key, mapped_action in keymap.items():
        if pressed[key]:
          action = mapped_action
          break
      else:
        if args.wait and not env._player.sleeping:
          continue
        else:
          action = 'noop'

    # Environment step.
    _, reward, done, _ = env.step(env.action_names.index(action))
    duration += 1
    
    # Lightweight profiling output
    if args.profile and frames_since_profile >= args.profile_frames:
      fps = clock.get_fps()
      rss_kb = _linux_rss_kb()
      if rss_kb >= 0:
        print(f'[PROFILE] fps={fps:.1f} rss={rss_kb/1024:.1f}MB step={env._step}')
      else:
        print(f'[PROFILE] fps={fps:.1f} rss=unknown step={env._step}')
      frames_since_profile = 0
    
    # Track reward for persistent display
    if reward:
      last_reward = reward
      last_reward_step = env._step
      reward_display_timer = reward_persistence

    # Track achievements.
    unlocked = {
        name for name, count in env._player.achievements.items()
        if count > 0 and name not in achievements}
    if unlocked:
      achievements |= unlocked
      # Only print to console if stats window is disabled
      if not args.stats:
        total = len(env._player.achievements.keys())
        for name in unlocked:
          print(f'Achievement ({len(achievements)}/{total}): {name}')
    
    if not args.stats:
      if env._step > 0 and env._step % 100 == 0:
        print(f'Time step: {env._step}')
      if reward:
        print(f'Reward: {reward}')
    
    if reward:
      return_ += reward

    # Episode end.
    if done and not was_done:
      was_done = True
      print('Episode done!')
      print('Duration:', duration)
      print('Return:', return_)
      if args.death == 'quit':
        running = False
      if args.death == 'reset':
        print('\nStarting a new episode.')
        obs = env.reset()
        achievements = set()
        was_done = False
        duration = 0
        return_ = 0
      if args.death == 'continue':
        pass

  pygame.quit()


if __name__ == '__main__':
  main()
