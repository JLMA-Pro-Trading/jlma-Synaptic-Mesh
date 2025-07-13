"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.peerCommand = peerCommand;
const commander_1 = require("commander");
function peerCommand() {
    const command = new commander_1.Command('peer');
    command.description('peer command');
    return command;
}
//# sourceMappingURL=peer.js.map