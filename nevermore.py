import random
from sys import platform


class Nevermore():
    random_item_ability = None

    shadowraze_damage = {0: 0, 1: 90, 2: 160, 3: 230, 4: 300}
    stack_bonus_damage = {0: 0, 1: 50, 2: 60, 3: 70, 4: 80}

    # 怨灵系带卷轴 (215)
    # 怨灵系带卷轴 (215) + 敏捷便鞋(135) + 贵族圆环(165) = 怨灵系带 (515)
    # 速度之靴 (500)
    # 手套 (500) + 精灵布袋 (450) = 假腿
    # 水晶剑 (2130)
    item_route = [{
        'enough_gold': 300,
        'items': ['item_magic_stick']
    }, {
        'enough_gold': 600,
        'items': ['item_boots']
    }, {
        'enough_gold': 350,
        'items': ['item_branches', 'item_branches', 'item_recipe_magic_wand']
    }, {
        'enough_gold': 350,
        'items': ['item_recipe_wraith_band']
    }, {
        'enough_gold': 650,
        'items': ['item_circlet', 'item_gauntlets', 'item_recipe_bracer']
    }, {
        'enough_gold': 650,
        'items': ['item_circlet', 'item_gauntlets', 'item_recipe_bracer']
    }]

    # special_bonus_attack_speed_20
    # special_bonus_movement_speed_3
    # special_bonus_unique_nevermore_2
    # special_bonus_cooldown_reduction_4
    routes = [
        'nevermore_necromastery', 'nevermore_shadowraze1', 'nevermore_shadowraze1', 'nevermore_necromastery',
        'nevermore_shadowraze1', 'nevermore_necromastery', 'nevermore_shadowraze1', 'nevermore_necromastery',
        'nevermore_dark_lord', 'special_bonus_spell_amplify_8', 'nevermore_requiem', 'nevermore_dark_lord',
        'nevermore_dark_lord', 'nevermore_dark_lord', 'special_bonus_unique_nevermore_3', 'nevermore_requiem',
        'nevermore_requiem', 'special_bonus_unique_nevermore_1', 'special_bonus_unique_nevermore_5'
    ]

    if platform == 'win32' or platform == 'darwin':
        routes[9] = 'special_bonus_spell_amplify_6'

    def __init__(self, running_mode):
        self.running_mode = running_mode
        # 出门装 2 树枝(50g), 净化药水(50g)， 大药膏(110), 敏捷便鞋(145), 贵族圆环(155)
        # self.init_item = ['item_circlet', 'item_slippers', 'item_flask', 'item_faerie_fire', 'item_faerie_fire']
        # random_items_num = random.randint(0, 100)
        # if random_items_num < 30:
        #     self.init_item = ['item_circlet', 'item_slippers', 'item_tango', 'item_faerie_fire', 'item_faerie_fire']

        if self.random_item_ability is None: # and self.running_mode != "self_eval":
            self.random_item_ability = 61 # random.randint(0, 100)
            # 技能学习增加随机性
            if self.random_item_ability >= 60:
                self.routes[0] = 'nevermore_shadowraze1'
                self.routes[1] = 'nevermore_necromastery'
                # self.init_item = [
                #     'item_circlet', 'item_faerie_fire', 'item_flask', 'item_enchanted_mango', 'item_enchanted_mango',
                #     'item_enchanted_mango'
                # ]
                # if random_items_num < 30:
                self.init_item = [
                    'item_flask', 'item_faerie_fire', 'item_flask', 'item_enchanted_mango', 'item_enchanted_mango',
                    'item_enchanted_mango', 'item_enchanted_mango'
                ]

                self.item_route = [{
                    'enough_gold': 300,
                    'items': ['item_magic_stick']
                }, {
                    'enough_gold': 600,
                    'items': ['item_boots']
                }, {
                    'enough_gold': 350,
                    'items': ['item_branches', 'item_branches', 'item_recipe_magic_wand']
                }, {
                    'enough_gold': 500,
                    'items': ['item_gauntlets', 'item_recipe_bracer']
                }, {
                    'enough_gold': 650,
                    'items': ['item_circlet', 'item_gauntlets', 'item_recipe_bracer']
                }, {
                    'enough_gold': 650,
                    'items': ['item_circlet', 'item_gauntlets', 'item_recipe_bracer']
                }]

            elif self.random_item_ability >= 50 and self.random_item_ability < 60 and self.running_mode != "ai_vs_ai":
                self.routes[5] = 'nevermore_requiem'
                self.routes[8] = 'nevermore_necromastery'
                self.routes[-3] = 'nevermore_dark_lord'

        r = random.randint(0, 1)
        if r == 1 or running_mode == "self_eval" or running_mode == "ai_vs_ai":
            self.init_item.append('item_ward_observer')

    def can_buy_new_item(self, used_gold, current_player):
        if len(self.item_route) == 0:
            return False
        gold = current_player.unreliable_gold + current_player.reliable_gold
        return gold - used_gold > self.item_route[0]['enough_gold']

    def need_item(self):
        return self.item_route.pop(0)

    def ability_route(self, ability_info):

        used_point = 0
        for key, x in ability_info.items():
            if key not in [5059, 5060]:
                used_point += x.level

        return self.routes[used_point]
