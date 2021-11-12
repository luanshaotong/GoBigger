import logging
import pytest
import uuid
from pygame.math import Vector2
import pygame
import random
import numpy as np
import cv2
import time

from .base_render import BaseRender
from gobigger.utils import FOOD_COLOR, THORNS_COLOR, SPORE_COLOR, PLAYER_COLORS, BACKGROUND, BLACK, WHITE
from gobigger.utils import FOOD_COLOR_GRAYSCALE, THORNS_COLOR_GRAYSCALE, SPORE_COLOR_GRAYSCALE, PLAYER_COLORS_GRAYSCALE, BACKGROUND_GRAYSCALE
from gobigger.utils import to_aliased_circle


class EnvRender(BaseRender):
    '''
    Overview:
        No need to use a new window, giving a global view and the view that each player can see
    '''
    def __init__(self, width, height, padding=(0,0), cell_size=10, 
                 scale_up_ratio=1.5, vision_x_min=100, vision_y_min=100, only_render=True):
        super(EnvRender, self).__init__(width, height, padding=padding, 
                                        cell_size=cell_size, only_render=only_render)
        self.scale_up_ratio = scale_up_ratio
        self.vision_x_min = vision_x_min
        self.vision_y_min = vision_y_min
        self.fill_all_time = 0
        self.transfer_rgb_to_features_all_time = 0
        self.get_overlap_all_time = 0
        self.get_clip_screen_all_time = 0
        self.get_rectangle_by_player_all_time = 0
        self.fill_count = 0
        self.fill_food_all_time = 0
        self.fill_thorns_all_time = 0
        self.fill_spore_all_time = 0
        self.fill_player_all_time = 0
        self.fill_array_all_time = 0
        self.flip_all_time = 0
        self.t_transfer_all_time = 0
        self.get_overlap_init_all_time = 0
        self.get_overlap_food_all_time = 0
        self.get_overlap_thorns_all_time = 0
        self.get_overlap_spore_all_time = 0
        self.get_overlap_clone_all_time = 0

    def set_obs_settings(self, obs_settings):
        self.with_spatial = obs_settings.get('with_spatial', True)
        self.with_speed = obs_settings.get('with_speed', False)
        self.with_all_vision = obs_settings.get('with_all_vision', False)

    def fill_food_balls(self, balls):
        if len(balls) == 0:
            return np.zeros((self.width, self.height), dtype=int)
        else:
            screen = pygame.Surface((self.width, self.height), depth=8)
            for ball in balls:
                pygame.draw.circle(screen, (1), ball.position, ball.radius)
            return np.rot90(np.fliplr(pygame.surfarray.array2d(screen)))

    def fill_spore_balls(self, balls):
        if len(balls) == 0:
            return np.zeros((self.width, self.height), dtype=int)
        else:
            screen = pygame.Surface((self.width, self.height), depth=8)
            for ball in balls:
                pygame.draw.circle(screen, (1), ball.position, ball.radius)
            return np.rot90(np.fliplr(pygame.surfarray.array2d(screen)))

    def fill_thorns_balls(self, balls):
        if len(balls) == 0:
            return np.zeros((self.width, self.height), dtype=int)
        else:
            screen = pygame.Surface((self.width, self.height), depth=8)
            for ball in balls:
                pygame.draw.polygon(screen, (1), to_aliased_circle(ball.position, ball.radius))
            return np.rot90(np.fliplr(pygame.surfarray.array2d(screen)))

    def fill_clone_balls(self, balls):
        if len(balls) == 0:
            return np.zeros((self.width, self.height), dtype=int)
        else:
            screen = pygame.Surface((self.width, self.height), depth=8)
            for ball in balls:
                pygame.draw.circle(screen, (1), ball.position, ball.radius)
            return np.rot90(np.fliplr(pygame.surfarray.array2d(screen)))

    def fill_all(self, food_balls, thorns_balls, spore_balls, players):
        # render all balls
        features = np.zeros((3+len(players), self.width, self.height))
        t_1 = time.time()
        features[len(players)] = self.fill_food_balls(food_balls)
        t_2 = time.time()
        features[len(players)+2] = self.fill_thorns_balls(thorns_balls)
        t_3 = time.time()
        features[len(players)+1] = self.fill_spore_balls(spore_balls)
        t_4 = time.time()
        for index, player in enumerate(players):
            features[int(player.name)] = self.fill_clone_balls(player.get_balls())
        t_5 = time.time()
        t_6 = time.time()
        self.fill_food_all_time += t_2-t_1
        self.fill_thorns_all_time += t_3-t_2
        self.fill_spore_all_time += t_4-t_3
        self.fill_player_all_time += t_5-t_4
        self.fill_array_all_time += t_6-t_5
        return features, [t_2-t_1, self.fill_food_all_time/self.fill_count,
                             t_3-t_2, self.fill_thorns_all_time/self.fill_count,
                             t_4-t_3, self.fill_spore_all_time/self.fill_count,
                             t_5-t_4, self.fill_player_all_time/self.fill_count,
                             t_6-t_5, self.fill_array_all_time/self.fill_count]

    def get_clip_screen(self, screen_data, rectangle):
        screen_data_clip = screen_data[:, rectangle[0]:rectangle[2], rectangle[1]:rectangle[3]]
        return screen_data_clip

    def get_rectangle_by_player(self, player):
        '''
        Multiples of the circumscribed matrix of the centroid
        '''
        centroid = player.cal_centroid()
        xs_max = 0
        ys_max = 0
        for ball in player.get_balls():
            direction_center = centroid - ball.position
            if abs(direction_center.x) + ball.radius > xs_max:
                xs_max = abs(direction_center.x) + ball.radius
            if abs(direction_center.y) + ball.radius > ys_max:
                ys_max = abs(direction_center.y) + ball.radius
        xs_max = max(xs_max, self.vision_x_min)
        ys_max = max(ys_max, self.vision_y_min)
        scale_up_len =  max(xs_max, ys_max)
        left_top_x = min(max(int(centroid.x - scale_up_len * self.scale_up_ratio), 0), 
                         max(int(self.width_full - scale_up_len * self.scale_up_ratio * 2), 0))
        left_top_y = min(max(int(centroid.y - scale_up_len * self.scale_up_ratio), 0),
                         max(int(self.height_full - scale_up_len * self.scale_up_ratio * 2), 0))
        right_bottom_x = min(int(left_top_x + scale_up_len * self.scale_up_ratio * 2), self.width_full)
        right_bottom_y = min(int(left_top_y + scale_up_len * self.scale_up_ratio * 2), self.height_full)
        rectangle = (left_top_x, left_top_y, right_bottom_x, right_bottom_y)
        return rectangle

    def get_overlap(self, rectangle, food_balls, thorns_balls, spore_balls, players):
        t1 = time.time()
        ret = {}
        food_count = 0
        thorns_count = 0
        spore_count = 0
        clone_count = 0
        food = 2500*[3*[None]]
        thorns = 20*[3*[None]]
        spore = 10000*[3*[None]]
        clone = 100*[5*[None]]
        t2 = time.time()
        for ball in food_balls:
            if ball.judge_in_rectangle(rectangle):
                # ret['food'].append({'position': tuple(ball.position), 'radius': ball.radius})
                # ret['food'].append([ball.position.x, ball.position.y, ball.radius])
                food[food_count][0] = ball.position.x
                food[food_count][1] = ball.position.y
                food[food_count][2] = ball.radius
                food_count += 1
        food = food[:food_count]
        ret['food'] = food

        t3 = time.time()
        for ball in thorns_balls:
            if ball.judge_in_rectangle(rectangle):
                # ret['thorns'].append({'position': tuple(ball.position), 'radius': ball.radius})
                # ret['thorns'].append([ball.position.x, ball.position.y, ball.radius])
                thorns[thorns_count][0] = ball.position.x
                thorns[thorns_count][1] = ball.position.y
                thorns[thorns_count][2] = ball.radius
                thorns_count += 1
        thorns = thorns[:thorns_count]
        ret['thorns'] = thorns

        t4 = time.time()
        for ball in spore_balls:
            if ball.judge_in_rectangle(rectangle):
                # ret['spore'].append([ball.position.x, ball.position.y, ball.radius])
                spore[spore_count][0] = ball.position.x
                spore[spore_count][1] = ball.position.y
                spore[spore_count][2] = ball.radius
                spore_count += 1
        spore = spore[:spore_count]
        ret['spore'] = spore

        t5 = time.time()
        for player in players:
            for ball in player.get_balls():
                if ball.judge_in_rectangle(rectangle):
                    # ret['clone'].append({'position': tuple(ball.position), 'radius': ball.radius, 
                    #                      'player': player.name, 'team': player.team_name})
                    # ret['clone'].append({ball.position.x, ball.position.y, ball.radius, 
                    #                      player.name, player.team_name})
                    clone[clone_count][0] = ball.position.x
                    clone[clone_count][1] = ball.position.y
                    clone[clone_count][2] = ball.radius
                    clone[clone_count][3] = player.name
                    clone[clone_count][4] = player.team_name
                    clone_count += 1
        clone = clone[:clone_count]
        ret['spore'] = clone
        t6 = time.time()

        return ret, [t2-t1, t3-t2, t4-t3, t5-t4, t6-t5]

    def get_overlap_wo_rectangle(self, food_balls, thorns_balls, spore_balls, players):
        ret = {'food': [], 'thorns': [], 'spore': [], 'clone': []}
        for ball in food_balls:
            ret['food'].append({'position': tuple(ball.position), 'radius': ball.radius})
        for ball in thorns_balls:
            ret['thorns'].append({'position': tuple(ball.position), 'radius': ball.radius})
        for ball in spore_balls:
            ret['spore'].append({'position': tuple(ball.position), 'radius': ball.radius})
        for player in players:
            for ball in player.get_balls():
                ret['clone'].append({'position': tuple(ball.position), 'radius': ball.radius, 
                                     'player': player.name, 'team': player.team_name})
        return ret

    def get_overlap_wo_rectangle_with_speed(self, food_balls, thorns_balls, spore_balls, players):
        ret = {'food': [], 'thorns': [], 'spore': [], 'clone': []}
        for ball in food_balls:
            ret['food'].append({'position': tuple(ball.position), 'radius': ball.radius})
        for ball in thorns_balls:
            ret['thorns'].append({'position': tuple(ball.position), 'radius': ball.radius, 'speed': tuple(ball.vel)})
        for ball in spore_balls:
            ret['spore'].append({'position': tuple(ball.position), 'radius': ball.radius, 'speed': tuple(ball.vel)})
        for player in players:
            for ball in player.get_balls():
                ret['clone'].append({'position': tuple(ball.position), 'radius': ball.radius, 'speed': tuple(ball.vel+ball.vel_last), 
                                     'player': player.name, 'team': player.team_name})
        return ret

    def get_overlap_with_speed(self, rectangle, food_balls, thorns_balls, spore_balls, players):
        ret = {'food': [], 'thorns': [], 'spore': [], 'clone': []}
        for ball in food_balls:
            if ball.judge_in_rectangle(rectangle):
                ret['food'].append({'position': tuple(ball.position), 'radius': ball.radius})
        for ball in thorns_balls:
            if ball.judge_in_rectangle(rectangle):
                ret['thorns'].append({'position': tuple(ball.position), 'radius': ball.radius, 'speed': tuple(ball.vel)})
        for ball in spore_balls:
            if ball.judge_in_rectangle(rectangle):
                ret['spore'].append({'position': tuple(ball.position), 'radius': ball.radius, 'speed': tuple(ball.vel)})
        for player in players:
            for ball in player.get_balls():
                if ball.judge_in_rectangle(rectangle):
                    ret['clone'].append({'position': tuple(ball.position), 'radius': ball.radius, 'speed': tuple(ball.vel+ball.vel_last), 
                                         'player': player.name, 'team': player.team_name})
        return ret

    def update_all(self, food_balls, thorns_balls, spore_balls, players):
        screen_data_all = None
        feature_layers = None
        overlap = None
        rectangle = None
        self.fill_count += 1
        t1 = time.time()
        if self.with_spatial:
            screen_data_all, t_f = self.fill_all(food_balls, thorns_balls, spore_balls, players)
        t2 = time.time()
        screen_data_players = {}

        if self.with_all_vision:
            for player in players:
                if player.name == '0':
                    if self.with_spatial:
                        screen_data_player = np.fliplr(screen_data_all)
                        screen_data_player = np.rot90(screen_data_player)
                        feature_layers = self.transfer_rgb_to_features(screen_data_player, player_num=len(players))
                    if self.with_speed:
                        overlap = self.get_overlap_wo_rectangle_with_speed(food_balls, thorns_balls, spore_balls, players)
                    else:
                        overlap = self.get_overlap_wo_rectangle(food_balls, thorns_balls, spore_balls, players)
                    screen_data_players[player.name] = {
                        'feature_layers': feature_layers,
                        'rectangle': rectangle,
                        'overlap': overlap,
                        'team_name': player.team_name,
                    }
                else:
                    screen_data_players[player.name] = {
                        'feature_layers': None,
                        'rectangle': None,
                        'overlap': None,
                        'team_name': player.team_name,
                    }
        else:
            get_rectangle_by_player_temp_time = 0
            get_clip_screen_temp_time = 0
            transfer_rgb_to_features_temp_time = 0
            flip_temp_time = 0
            get_overlap_temp_time = 0
            t_transfer_temp_time = 0
            get_overlap_init_temp_time = 0
            get_overlap_food_temp_time = 0
            get_overlap_thorns_temp_time = 0
            get_overlap_spore_temp_time = 0
            get_overlap_clone_temp_time = 0

            for player in players:
                t4 = time.time()
                rectangle = self.get_rectangle_by_player(player)
                t5 = time.time()
                if self.with_spatial:
                    feature_layers = self.get_clip_screen(screen_data_all, rectangle=rectangle)
                    t6 = time.time()
                    t9 = time.time()
                    t_transfer = 0
                t7 = time.time()
                if self.with_speed:
                    overlap = self.get_overlap_with_speed(rectangle, food_balls, thorns_balls, spore_balls, players)
                else:
                    overlap, t_overlap = self.get_overlap(rectangle, food_balls, thorns_balls, spore_balls, players)
                t8 = time.time()
                screen_data_players[player.name] = {
                    'feature_layers': feature_layers,
                    'rectangle': rectangle,
                    'overlap': overlap,
                    'team_name': player.team_name,
                }
                get_rectangle_by_player_temp_time += t5-t4
                get_clip_screen_temp_time += t6-t5
                transfer_rgb_to_features_temp_time += t7-t9
                flip_temp_time += t9 - t6
                get_overlap_temp_time += t8-t7
                t_transfer_temp_time += t_transfer
                get_overlap_init_temp_time += t_overlap[0]
                get_overlap_food_temp_time += t_overlap[1]
                get_overlap_thorns_temp_time += t_overlap[2]
                get_overlap_spore_temp_time += t_overlap[3]
                get_overlap_clone_temp_time += t_overlap[4]

            self.get_rectangle_by_player_all_time += get_rectangle_by_player_temp_time
            self.get_clip_screen_all_time += get_clip_screen_temp_time
            self.transfer_rgb_to_features_all_time += transfer_rgb_to_features_temp_time
            self.flip_all_time += flip_temp_time
            self.get_overlap_all_time += get_overlap_temp_time
            self.t_transfer_all_time += t_transfer_temp_time
            self.get_overlap_init_all_time += get_overlap_init_temp_time
            self.get_overlap_food_all_time += get_overlap_food_temp_time
            self.get_overlap_thorns_all_time += get_overlap_thorns_temp_time
            self.get_overlap_spore_all_time += get_overlap_spore_temp_time
            self.get_overlap_clone_all_time += get_overlap_clone_temp_time

        self.fill_all_time += t2-t1
        t = [self.fill_count, t2-t1, self.fill_all_time/self.fill_count,
             get_rectangle_by_player_temp_time, self.get_rectangle_by_player_all_time/self.fill_count,
             get_clip_screen_temp_time, self.get_clip_screen_all_time/self.fill_count,
             transfer_rgb_to_features_temp_time, self.transfer_rgb_to_features_all_time/self.fill_count,
             t_transfer_temp_time, self.t_transfer_all_time/self.fill_count,
             flip_temp_time, self.flip_all_time/self.fill_count,
             get_overlap_temp_time, self.get_overlap_all_time/self.fill_count, *t_f,
             get_overlap_init_temp_time, self.get_overlap_init_all_time/self.fill_count,
             get_overlap_food_temp_time, self.get_overlap_food_all_time/self.fill_count,
             get_overlap_thorns_temp_time, self.get_overlap_thorns_all_time/self.fill_count,
             get_overlap_spore_temp_time, self.get_overlap_spore_all_time/self.fill_count,
             get_overlap_clone_temp_time, self.get_overlap_clone_all_time/self.fill_count,
             ]
        return screen_data_all, screen_data_players, t

    def render_all_balls_colorful(self, screen, food_balls, thorns_balls, spore_balls, players, player_num_per_team):
        # render all balls
        for ball in food_balls:
            pygame.draw.circle(screen, FOOD_COLOR, ball.position, ball.radius)
        for ball in thorns_balls:
            pygame.draw.polygon(screen, THORNS_COLOR, to_aliased_circle(ball.position, ball.radius))
        for ball in spore_balls:
            pygame.draw.circle(screen, SPORE_COLOR, ball.position, ball.radius)
        player_name_size = {}
        for index, player in enumerate(players):
            for ball in player.get_balls():
                pygame.draw.circle(screen, PLAYER_COLORS[int(ball.team_name)][0], ball.position, ball.radius)
                font_size = int(ball.radius/1.6)
                font = pygame.font.SysFont('arial', max(font_size, 8), True)
                txt = font.render('{}'.format(chr(int(ball.owner)%player_num_per_team+65)), True, WHITE)
                txt_rect = txt.get_rect(center=(ball.position.x, ball.position.y))
                screen.blit(txt, txt_rect)
            player_name_size[player.name] = player.get_total_size()
        return screen, player_name_size

    def render_leaderboard_colorful(self, screen, team_name_size, player_name_size, player_num_per_team):
        team_name_size = sorted(team_name_size.items(), key=lambda d: d[1], reverse=True)
        start = 10
        for index, (team_name, size) in enumerate(team_name_size):
            start += 20
            font = pygame.font.SysFont('arial', 16, True)
            fps_txt = font.render('{} : {:.3f}'.format(team_name, size), True, PLAYER_COLORS[int(team_name)][0])
            screen.blit(fps_txt, (self.width+20, start))
            start += 20
            font = pygame.font.SysFont('arial', 14, True)
            for j in range(player_num_per_team):
                player_name = str(int(team_name)*player_num_per_team+j)
                player_size = player_name_size[player_name]
                fps_txt = font.render('  {} : {:.3f}'.format(chr(int(player_name)%player_num_per_team+65), player_size), True, PLAYER_COLORS[int(team_name)][0])
                screen.blit(fps_txt, (self.width+20, start))
                start += 20
        return screen

    def get_tick_all_colorful(self, food_balls, thorns_balls, spore_balls, players, partial_size=300, player_num_per_team=3, 
                              bar_width=150, team_name_size=None):
        screen_all = pygame.Surface((self.width+bar_width, self.height))
        screen_all.fill(BACKGROUND)
        pygame.draw.line(screen_all, BLACK, (self.width+1, 0), (self.width+1, self.height), width=3)
        screen_all, player_name_size = self.render_all_balls_colorful(screen_all, food_balls, thorns_balls, spore_balls, players, 
                                                                      player_num_per_team=player_num_per_team)
        screen_all = self.render_leaderboard_colorful(screen_all, team_name_size, player_name_size, player_num_per_team=player_num_per_team)
        screen_data_all = pygame.surfarray.array3d(screen_all)
        screen_data_players = {}
        for player in players:
            rectangle = self.get_rectangle_by_player(player)
            screen_data_player = self.get_clip_screen(screen_data_all, rectangle=rectangle)
            screen_data_player = cv2.resize(np.rot90(np.fliplr(cv2.cvtColor(screen_data_player, cv2.COLOR_RGB2BGR))), (partial_size, partial_size))
            screen_data_players[player.name] = screen_data_player
        screen_data_all = np.rot90(np.fliplr(cv2.cvtColor(screen_data_all, cv2.COLOR_RGB2BGR)))
        return screen_data_all, screen_data_players

    def transfer_rgb_to_features(self, rgb, player_num=12):
        '''
        Overview:
            If player_num == 12, then the features list will contain 15 elements(12 player + food + spore + thorns)
        '''
        # features = []
        # assert len(rgb.shape) == 2
        # h, w = rgb.shape
        # for i in range(player_num):
        #     features.append((rgb==PLAYER_COLORS_GRAYSCALE[i]).astype(int))
        # features.append((rgb==FOOD_COLOR_GRAYSCALE).astype(int))
        # features.append((rgb==SPORE_COLOR_GRAYSCALE).astype(int))
        # features.append((rgb==THORNS_COLOR_GRAYSCALE).astype(int))

        h, w = rgb.shape
        total_len = player_num + 3
        if not hasattr(self, 'to_equal_arr'):
            arr = []
            for i in range(player_num):
                arr.append(PLAYER_COLORS_GRAYSCALE[i])
            arr.append(FOOD_COLOR_GRAYSCALE)
            arr.append(SPORE_COLOR_GRAYSCALE)
            arr.append(THORNS_COLOR_GRAYSCALE)
            self.to_equal_arr = np.array(arr, dtype=np.int)
            self.to_equal_arr = self.to_equal_arr.reshape(total_len, 1, 1)
        t_transfer_1 = time.time()
        rgb = rgb.reshape(1, h, w)
        rgb = np.repeat(rgb, total_len, axis=0)
        features = (rgb == self.to_equal_arr).astype(int)
        t_transfer_2 = time.time()

        return features, t_transfer_2 - t_transfer_1

    def show(self):
        raise NotImplementedError

    def close(self):
        pygame.quit()
