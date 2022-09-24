from discord.ext import commands
import discord
from log import get_logger

class DreamBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.members = True
        super().__init__(command_prefix='d!', intents=intents)
        self.logger = get_logger(__name__)

    async def on_ready(self):
        self.logger.info(f'Logged in as {self.user.name} ({self.user.id})')
        await self.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name='Vendo as coisas degen que vocês geram...'))
        self.load_extension('stablecog')

    async def close(self):
        await self._bot.close()

