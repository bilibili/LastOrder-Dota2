from dotaservice.protos.dota_gcmessages_common_bot_script_pb2 import CMsgBotWorldState
from gym_env.feature_processors.enums import *

COURIER_STATE_INIT = -1
COURIER_STATE_IDLE = 0
COURIER_STATE_AT_BASE = 1
COURIER_STATE_MOVING = 2
COURIER_STATE_DELIVERING_ITEMS = 3
COURIER_STATE_RETURNING_TO_BASE = 4
COURIER_STATE_DEAD = 5

COURIER_ACTION_BURST = 5
COURIER_ACTION_ENEMY_SECRET_SHOP = 7
COURIER_ACTION_RETURN = 0
COURIER_ACTION_SECRET_SHOP = 1
COURIER_ACTION_SIDE_SHOP = 8
COURIER_ACTION_SIDE_SHOP2 = 9
COURIER_ACTION_TAKE_STASH_ITEMS = 3
COURIER_ACTION_TAKE_AND_TRANSFER_ITEMS = 6
COURIER_ACTION_TRANSFER_ITEMS = 4
COURIER_ACTION_RETURN_STASH_ITEMS = 2


class BasicAction():
    """
    自定义脚本动作
    """

    def __init__(self, player_id):
        self.player_id = player_id

    def update(self, player_history_info, self_courier_history):
        self.player_history_info = player_history_info
        self.self_courier_history = self_courier_history

    def get_action_message(self, action_pb, cur_dota_time, sync_key):
        # 0 for first radiant , 5 for first dire
        action_pb.player = self.player_id
        actions_pb = CMsgBotWorldState.Actions(actions=[action_pb], dota_time=cur_dota_time, extraData=sync_key)
        return actions_pb

    def get_extra_message(self, actions_pb):
        for action_pb in actions_pb:
            action_pb.player = self.player_id
        actions = CMsgBotWorldState.Actions(actions=actions_pb)
        return actions

    def train_ability(self, name):
        action_pb = CMsgBotWorldState.Action()
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_TRAIN_ABILITY')
        action_pb.trainAbility.ability = name
        return action_pb

    def use_ability(self, slot):
        action_pb = CMsgBotWorldState.Action()
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_NO_TARGET')
        action_pb.cast.abilitySlot = slot
        return action_pb

    def attack(self, u):
        action_pb = CMsgBotWorldState.Action()
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_ATTACK_TARGET')
        action_pb.attackTarget.target = u['handle']
        action_pb.attackTarget.once = True
        return action_pb

    def deny(self, u):
        return self.attack(u)

    def enemy(self, u):
        return self.attack(u)

    # 走向起始点
    def move(self, x, y, z, directly=True):
        action_pb = CMsgBotWorldState.Action()
        if directly:
            action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_MOVE_DIRECTLY')
            action_pb.moveDirectly.location.x = x
            action_pb.moveDirectly.location.y = y
            action_pb.moveDirectly.location.z = z
        else:
            action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_MOVE_TO_POSITION')
            action_pb.moveToLocation.location.x = x
            action_pb.moveToLocation.location.y = y
            action_pb.moveToLocation.location.z = z
        return action_pb

    # 原地不动
    def do_nothing(self):
        action_pb = CMsgBotWorldState.Action()
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_NONE')
        return action_pb

    def stop(self):
        action_pb = CMsgBotWorldState.Action()
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_STOP')
        return action_pb

    # 聊天
    def chat(self, message):
        action_pb = CMsgBotWorldState.Action()
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('ACTION_CHAT')
        action_pb.chat.message = message
        action_pb.chat.to_allchat = True
        return action_pb

    # 嘲讽
    def taunt(self):
        return self.chat('?')

    # 购买物品
    def buy(self, name):
        action_pb = CMsgBotWorldState.Action()
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_PURCHASE_ITEM')
        action_pb.purchaseItem.item_name = name
        return action_pb

    # 使用物品
    def use_item(self, slot):
        action_pb = CMsgBotWorldState.Action()
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_NO_TARGET')
        # slot 大于等于 0 为技能， 小于 0 的为物品
        action_pb.cast.abilitySlot = -slot - 1
        return action_pb

    def use_item_on(self, slot, target_id):
        action_pb = CMsgBotWorldState.Action()
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_TARGET')
        # slot 大于等于 0 为技能， 小于 0 的为物品
        action_pb.castTarget.abilitySlot = -slot - 1
        action_pb.castTarget.target = target_id
        return action_pb

    def use_item_on_tree(self, slot, tree):
        action_pb = CMsgBotWorldState.Action()
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_TARGET_TREE')
        # slot 大于等于 0 为技能， 小于 0 的为物品
        action_pb.castTree.abilitySlot = -slot - 1
        action_pb.castTree.tree = tree
        return action_pb

    def use_item_on_postion(self, slot, location):
        action_pb = CMsgBotWorldState.Action()
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('DOTA_UNIT_ORDER_CAST_POSITION')
        # slot 大于等于 0 为技能， 小于 0 的为物品
        action_pb.castLocation.abilitySlot = -slot - 1
        action_pb.castLocation.location.x = location.x
        action_pb.castLocation.location.y = location.y
        action_pb.castLocation.location.z = location.z
        return action_pb

    def courier_go_home(self):
        action_pb = CMsgBotWorldState.Action()
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('ACTION_COURIER')
        # action_pb.courier.unit
        action_pb.courier.action = COURIER_ACTION_RETURN
        return action_pb

    def take_item_by_courier(self):
        action_pb = CMsgBotWorldState.Action()
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('ACTION_COURIER')
        # action_pb.courier.unit
        # action_pb.courier.courier = courier_id
        action_pb.courier.action = COURIER_ACTION_TAKE_AND_TRANSFER_ITEMS
        return action_pb

    def swap_item(self, s1, s2):
        action_pb = CMsgBotWorldState.Action()
        action_pb.actionType = CMsgBotWorldState.Action.Type.Value('ACTION_SWAP_ITEMS')
        action_pb.swapItems.slot_a = s1
        action_pb.swapItems.slot_b = s2
        return action_pb

    def find_item_slot_by_id(self, item_id, in_equipment=False):
        if len(self.player_history_info) == 0 or len(self.player_history_info[-1].items) == 0:
            return None

        items = self.player_history_info[-1].items

        for item in items:
            if item.ability_id == item_id: # and item['slot'] <= 5:
                if not in_equipment:
                    return item.slot
                elif in_equipment and item.slot <= 5:
                    return item.slot
        return None

    def find_item_slot_by_id_in_cd(self, item_id, in_equipment=False):
        if len(self.player_history_info) == 0 or len(self.player_history_info[-1].items) == 0:
            return None

        items = self.player_history_info[-1].items
        for item in items:
            if item.ability_id == item_id: # and item['slot'] <= 5:
                if not in_equipment:
                    return item.slot
                elif in_equipment and item.slot <= 5:
                    if item.cooldown_remaining == 0:
                        return item.slot
        return None

    def find_item_by_slot_courier(self, item_id):
        if self.self_courier_history[-1] is None or len(self.self_courier_history[-1].items) == 0:
            return None

        items = self.self_courier_history[-1].items
        for item in items:
            if item.ability_id == item_id:
                return item.slot
        return None

    def where_items(self, item_id):
        res = 0
        if len(self.player_history_info) == 0 or len(self.player_history_info[-1].items) == 0:
            return res

        items = self.player_history_info[-1].items

        for item in items:
            if item.ability_id == item_id:
                res += item.charges

        if self.self_courier_history[-1] is None or len(self.self_courier_history[-1].items) == 0:
            return res

        items = self.self_courier_history[-1].items
        for item in items:
            if item.ability_id == item_id:
                res += item.charges

        return res

    def eat_faerie_fire(self, slot=None):
        slot = self.find_item_slot_by_id(ITEM_FAERIE_FIRE_ID, in_equipment=True) if slot is None else slot
        return self.use_item(slot)

    def eat_nearest_tree(self, slot=None):
        slot = self.find_item_slot_by_id(ITEM_TANGO_ID, in_equipment=True) if slot is None else slot
        return self.use_item(slot)

    def eat_tree(self, tree_id, slot=None):
        slot = self.find_item_slot_by_id(ITEM_TANGO_ID, in_equipment=True) if slot is None else slot
        return self.use_item_on_tree(slot=slot, tree=tree_id)

    def drink_flask(self, slot=None):
        slot = self.find_item_slot_by_id(ITEM_FLASK_ID, in_equipment=True) if slot is None else slot
        return self.use_item_on(slot, self.player_history_info[-1].handle)

    def eat_mango(self, slot=None):
        slot = self.find_item_slot_by_id(ITEM_MANGO_ID, in_equipment=True) if slot is None else slot
        return self.use_item_on(slot, self.player_history_info[-1].handle)

    def drink_clarity(self, slot=None):
        slot = self.find_item_slot_by_id(ITEM_CLARITY_ID, in_equipment=True) if slot is None else slot
        return self.use_item_on(slot, self.player_history_info[-1].handle)

    def drink_bottle(self, slot=None):
        slot = self.find_item_slot_by_id(ITEM_BOTTLE_ID, in_equipment=True) if slot is None else slot
        return self.use_item_on(slot, self.player_history_info[-1].handle)

    def plant(self, slot=None):
        slot = self.find_item_slot_by_id(ITEM_BRANCH_ID, in_equipment=True) if slot is None else slot
        loc = self.player_history_info[-1].location
        vec = CMsgBotWorldState.Vector(x=loc.x + 10, y=loc.y + 10, z=loc.z)
        return self.use_item_on_postion(slot, vec)

    def use_tp(self, slot=None):
        # TODO 这里写的是夜宴一塔位置，但天辉也能用
        return self.use_item_on_postion(TP_SLOT, CMsgBotWorldState.Vector(x=523, y=651, z=256))

    def use_magic_stick(self, slot=None):
        # 大小魔棒
        if slot is None:
            slot = self.find_item_slot_by_id(ITEM_MAGIC_STICK_ID, in_equipment=True)

        if slot is None:
            slot = self.find_item_slot_by_id(ITEM_MAGIC_WAND_ID, in_equipment=True)

        return self.use_item(slot)

    def could_buy(self, name, used_gold=0):
        gold = self.player_history_info[-1].unreliable_gold + self.player_history_info[-1].reliable_gold - used_gold
        if name == 'item_flask':
            return gold >= 110
        elif name == 'item_clarity':
            return gold >= 50
        elif name == 'item_enchanted_mango':
            return gold >= 70
        elif name == 'item_ward_sentry':
            return gold >= 75

        return False

    def stash_item_need_take(self):
        if len(self.player_history_info) == 0 or len(self.player_history_info[-1].items) == 0:
            return False

        has_flask = False
        has_other_items = False
        mango_count = 0

        for item in self.player_history_info[-1].items:
            if item.slot >= 9 and item.slot < 15:
                if item.ability_id == ITEM_FLASK_ID:
                    has_flask = True
                elif item.ability_id == ITEM_MANGO_ID:
                    mango_count += item.charges
                elif item.ability_id == ITEM_WARD_ID or item.ability_id == ITEM_WARD_SENTRY_ID:
                    pass
                else:
                    has_other_items = True

        if len(self.self_courier_history) > 0:
            for item in self.self_courier_history[-1].items:
                if item.slot >= 0 and item.slot < 6:
                    if item.ability_id == ITEM_FLASK_ID:
                        has_flask = True
                    elif item.ability_id == ITEM_MANGO_ID:
                        mango_count += item.charges
                    elif item.ability_id == ITEM_WARD_ID or item.ability_id == ITEM_WARD_SENTRY_ID:
                        pass
                    else:
                        has_other_items = True

        return has_other_items or has_flask or mango_count >= 2
