'''
                     Copyright Oliver Kowalke 2018.
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import asyncio
import logging
import os
import subprocess

from dotenv import load_dotenv, find_dotenv
from pathlib import Path


def generate(offapi_p, t, n, template_p, outdir_p):
    cmd = 'src/data/generate %s %s %s %s -env:URE_MORE_TYPES=\'file://%s\' \'uno:socket,host=localhost,port=2083;urp;StarOffice.ServiceManager\'' \
        % (str(t), str(n), str(template_p), str(outdir_p), str(offapi_p))
    subprocess.check_call(cmd, shell=True)


async def async_exec(cmd):
    create = asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE)
    proc = await create
    try:
        await proc.wait()
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()


async def process_pdf(pdf_p):
    jpg_p = pdf_p.with_suffix('.jpg')
    await async_exec('convert -density 300 %s %s' % (str(pdf_p), str(jpg_p)))
    await async_exec('src/data/trim --file %s' % str(jpg_p))
    pdf_p.unlink()


def process_pdfs(raw_p, chunk_size):
    loop = asyncio.get_event_loop()
    pdfs_p = list(raw_p.glob('*.pdf'))
    assert len(pdfs_p) == len(list(raw_p.glob('*.csv')))
    for chunk in [pdfs_p[i:i+chunk_size] for i in range(0, len(pdfs_p), chunk_size)]:
        tasks = []
        for pdf_p in chunk:
            tasks.append(loop.create_task(process_pdf(pdf_p)))
        loop.run_until_complete(asyncio.wait(tasks))
    loop.close()


def main():
    env_path = find_dotenv()
    load_dotenv(dotenv_path=env_path, verbose=True)

    raw_p = Path(os.environ.get('PATH_RAW')).resolve()
    offapi_p = Path(os.environ.get('PATH_OFFAPI')).resolve()
    template_p = Path(os.environ.get('PATH_OFFTMPL')).resolve().as_uri()
    data_size = int(os.environ.get('DATA_SIZE'))
    chunk_size = int(os.environ.get('CHUNK_SIZE'))

    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    logger.info('generate raw data')
    if not raw_p.exists():
        raw_p.mkdir(parents=True)
    generate(offapi_p, chunk_size, data_size, template_p, raw_p)
    logger.info('post processing raw data')
    process_pdfs(raw_p, chunk_size)


if __name__ == '__main__':
    main()
