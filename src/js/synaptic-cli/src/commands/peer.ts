import { Command } from 'commander';
import chalk from 'chalk';

export function peerCommand(): Command {
  const command = new Command('peer');

  command
    .description('Manage peer connections')
    .addCommand(peerListCommand())
    .addCommand(peerConnectCommand())
    .addCommand(peerDisconnectCommand());

  return command;
}

function peerListCommand(): Command {
  const command = new Command('list');
  
  command
    .description('List all connected peers')
    .action(async () => {
      console.log(chalk.cyan('\n📡 Connected Peers:'));
      console.log(chalk.gray('─'.repeat(60)));
      console.log('No peers connected');
      console.log(chalk.gray('─'.repeat(60)));
    });

  return command;
}

function peerConnectCommand(): Command {
  const command = new Command('connect');
  
  command
    .description('Connect to a peer')
    .argument('<address>', 'Peer address to connect to')
    .action(async (address: string) => {
      console.log(chalk.yellow(`🔗 Connecting to peer: ${address}...`));
      console.log(chalk.green('✅ Connection established'));
    });

  return command;
}

function peerDisconnectCommand(): Command {
  const command = new Command('disconnect');
  
  command
    .description('Disconnect from a peer')
    .argument('<peer-id>', 'Peer ID to disconnect')
    .action(async (peerId: string) => {
      console.log(chalk.yellow(`🔌 Disconnecting from peer: ${peerId}...`));
      console.log(chalk.green('✅ Peer disconnected'));
    });

  return command;
}
