import asyncio
import traceback
from typing import List

import aiohttp
import uvloop
from openpyxl import load_workbook
from openpyxl.worksheet import Worksheet


class DataConfiguration:

    __slots__ = ['url', 'row']

    def __init__(self, url: str, row: int) -> None:
        self.url = url
        self.row = row


class CrawledDataConfiguration:

    __slots__ = ['html', 'row']

    def __init__(self, html: str, row: int) -> None:
        self.html = html
        self.row = row


def read_data(ws: Worksheet) -> List[DataConfiguration]:
    data = [
        DataConfiguration(ws[f'D{i}'].value, int(ws[f'A{i}'].value) + 2)
        for i in range(2, ws.max_row + 1)
    ]
    return data

async def send_request(data: DataConfiguration) -> CrawledDataConfiguration:
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(verify_ssl=False)
    ) as session:
        async with session.get(data.url) as res:
            html = await res.text()
            return CrawledDataConfiguration(html, data.row)

async def crawling_files(file_name):
    wb = load_workbook(file_name)
    ws = wb.worksheets[0]
    data_configurations = read_data(ws)
    requests = [
        send_request(data_configuration)
        for data_configuration in data_configurations
    ]
    crawled_data_configurations = await asyncio.gather(*requests[:10])
    for crawled_data_configuration in crawled_data_configurations:
        ws.cell(
            row=crawled_data_configuration.row,
            column=5,
            value=crawled_data_configuration.html
        )
    wb.save(file_name)

async def main(loop):
    try:
        file_names = [
            'competition_stances_uniqueUrl.xlsx',
            'train_stances_uniqueUrl.xlsx'
        ]
        for file_name in file_names:
            await crawling_files(file_name)
    except Exception as e:
        traceback.print_stack()
        print('-------------------------------')
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(loop))
    loop.close()
