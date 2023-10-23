# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import serial
import time

name = time.strftime("%Y%m%d-%H%M%S")
start = time.time()
duration_sec = 60

# Read serial fast > 100 kHz and write it to a file.
with serial.Serial('/dev/ttyACM0', 115200, timeout=1) as ser:
        with open(name + ".txt", "w") as f:
            buffer = []
            while True:
                try:
                    waiting = ser.in_waiting
                    buffer += [chr(c) for c in ser.read(waiting)]
                    if len(buffer) > 50:
                        f.write("".join(buffer))
                        buffer.clear()
                    #if time.time() - start > duration_sec:
                     #   break
                except Exception as e:
                    print("Error", e)

            f.write("".join(buffer))

