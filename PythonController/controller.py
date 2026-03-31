import argparse
import socket
from dataclasses import dataclass


ACTION_ORDER = ["LEFT", "RIGHT", "DOWN", "SPEED", "JUMP"]

OBS_BRICK = 22
OBS_QUESTION_BLOCK = 24
OBS_USED_BLOCK = 30


@dataclass
class ElementPos:
	type_id: int
	x: float
	y: float


@dataclass
class LevelBlock:
	x: int
	y: int
	tile_id: int


@dataclass
class LevelData:
	width: int
	height: int
	blocks: list[LevelBlock]


@dataclass
class StepObservation:
	step: int
	mario_x: float
	mario_y: float
	vel_x: float
	vel_y: float
	mode: int
	on_ground: bool
	may_jump: bool
	can_jump_higher: bool
	remaining_time: int
	completion: float
	status: str
	enemies: list[ElementPos]
	sprites: list[ElementPos]
	scene_width: int
	scene_height: int
	scene_tiles: list[int]


class MarioPythonController:
	def __init__(self) -> None:
		self.level_data: LevelData | None = None

	def set_level_data(self, level_data: LevelData) -> None:
		self.level_data = level_data

	def choose_actions(self, obs: StepObservation) -> list[bool]:
		actions = [False] * len(ACTION_ORDER)

		actions[1] = True   # RIGHT
		actions[3] = True   # SPEED

		# Periodic short hops while grounded are a simple baseline behavior.
		if obs.on_ground and obs.step % 18 in (0, 1):
			actions[4] = True

		# Extra example: jump if a block-like obstacle is very close in front.
		if has_block_ahead(obs):
			actions[4] = True

		return actions


def parse_bool(value: str) -> bool:
	return value.lower() in ("1", "true", "t", "yes", "y")


def parse_step(parts: list[str]) -> StepObservation:
	if len(parts) < 13:
		raise ValueError(f"STEP message has {len(parts)} fields, expected at least 13")

	enemies_raw = parts[13] if len(parts) > 13 else "-"
	sprites_raw = parts[14] if len(parts) > 14 else "-"
	scene_raw = parts[15] if len(parts) > 15 else "-"
	scene_w, scene_h, scene_tiles = parse_scene_grid(scene_raw)

	return StepObservation(
		step=int(parts[1]),
		mario_x=float(parts[2]),
		mario_y=float(parts[3]),
		vel_x=float(parts[4]),
		vel_y=float(parts[5]),
		mode=int(parts[6]),
		on_ground=parse_bool(parts[7]),
		may_jump=parse_bool(parts[8]),
		can_jump_higher=parse_bool(parts[9]),
		remaining_time=int(parts[10]),
		completion=float(parts[11]),
		status=parts[12],
		enemies=parse_element_positions(enemies_raw),
		sprites=parse_element_positions(sprites_raw),
		scene_width=scene_w,
		scene_height=scene_h,
		scene_tiles=scene_tiles,
	)


def parse_level(parts: list[str]) -> LevelData:
	if len(parts) < 4:
		raise ValueError(f"LEVEL message has {len(parts)} fields, expected at least 4")

	width = int(parts[1])
	height = int(parts[2])
	blocks_raw = parts[3]

	blocks: list[LevelBlock] = []
	if blocks_raw and blocks_raw != "-":
		for item in blocks_raw.split(";"):
			if not item:
				continue
			coords = item.split(",")
			if len(coords) != 3:
				continue
			try:
				blocks.append(LevelBlock(x=int(coords[0]), y=int(coords[1]), tile_id=int(coords[2])))
			except ValueError:
				continue
	print(blocks)
	return LevelData(width=width, height=height, blocks=blocks)


def parse_element_positions(raw: str) -> list[ElementPos]:
	if not raw or raw == "-":
		return []

	parsed: list[ElementPos] = []
	for item in raw.split(";"):
		if not item:
			continue
		parts = item.split(",")
		if len(parts) != 3:
			continue
		try:
			parsed.append(
				ElementPos(
					type_id=int(float(parts[0])),
					x=float(parts[1]),
					y=float(parts[2]),
				)
			)
		except ValueError:
			continue
	return parsed


def parse_scene_grid(raw: str) -> tuple[int, int, list[int]]:
	if not raw or raw == "-" or ":" not in raw:
		return 0, 0, []

	dims, data = raw.split(":", maxsplit=1)
	if "x" not in dims:
		return 0, 0, []

	try:
		w_str, h_str = dims.split("x", maxsplit=1)
		width = int(w_str)
		height = int(h_str)
	except ValueError:
		return 0, 0, []

	if not data:
		return width, height, []

	tiles: list[int] = []
	for value in data.split(","):
		if not value:
			continue
		try:
			tiles.append(int(value))
		except ValueError:
			tiles.append(0)

	return width, height, tiles


def has_block_ahead(obs: StepObservation) -> bool:
	if obs.scene_width <= 0 or obs.scene_height <= 0:
		return False

	center_x = obs.scene_width // 2
	center_y = obs.scene_height // 2
	target_positions = [
		(center_x + 1, center_y),
		(center_x + 1, center_y - 1),
		(center_x + 2, center_y),
		(center_x + 2, center_y - 1),
	]

	for tx, ty in target_positions:
		if tx < 0 or ty < 0 or tx >= obs.scene_width or ty >= obs.scene_height:
			continue
		idx = ty * obs.scene_width + tx
		if idx >= len(obs.scene_tiles):
			continue
		tile = obs.scene_tiles[idx]
		if tile in (OBS_BRICK, OBS_QUESTION_BLOCK, OBS_USED_BLOCK):
			return True

	return False


def actions_to_line(actions: list[bool]) -> str:
	return " ".join("1" if a else "0" for a in actions)


def serve(host: str, port: int) -> None:
	controller = MarioPythonController()

	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
		server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		server.bind((host, port))
		server.listen(1)
		print(f"Python controller listening on {host}:{port}")

		while True:
			conn, addr = server.accept()
			with conn:
				print(f"Java framework connected from {addr[0]}:{addr[1]}")
				reader = conn.makefile("r", encoding="utf-8", newline="\n")
				writer = conn.makefile("w", encoding="utf-8", newline="\n")

				while True:
					line = reader.readline()
					if not line:
						print("Java side closed connection")
						break

					parts = line.strip().split("\t")
					if not parts:
						continue

					tag = parts[0]
					if tag == "HELLO":
						print("Handshake received:", line.strip())
						continue

					if tag == "LEVEL":
						level_data = parse_level(parts)
						controller.set_level_data(level_data)
						print(
							f"Level received: {level_data.width}x{level_data.height}, "
							f"tracked blocks={len(level_data.blocks)}"
						)
						continue

					if tag == "STEP":
						obs = parse_step(parts)
						actions = controller.choose_actions(obs)
						writer.write(actions_to_line(actions) + "\n")
						writer.flush()
						continue

					if tag == "END":
						print("Game ended:", line.strip())
						break

					print("Unknown message:", line.strip())


def main() -> None:
	parser = argparse.ArgumentParser(description="Python-side Mario controller server")
	parser.add_argument("--host", default="127.0.0.1")
	parser.add_argument("--port", type=int, default=5050)
	args = parser.parse_args()

	serve(args.host, args.port)


if __name__ == "__main__":
	main()
