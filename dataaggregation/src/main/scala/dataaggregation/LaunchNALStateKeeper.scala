package dataaggregation

import akka.actor._

import scala.collection.mutable

import sp.EricaEventLogger._
import sp.gPubSub.API_Data.EricaEvent

object LaunchNALStateKeeper extends App {
  val system = ActorSystem("DataAggregation")

  val evHandler = new NALStateKeeper
  system.actorOf(Props(new Logger(evHandler)), "EricaEventLogger")
}

class NALStateKeeper extends RecoveredEventHandler {

  val patientsAtNAL = mutable.Set[Int]() // set of CareContactIds

  override def handleEvent(ev: EricaEvent) = {
    println("patients at NAL: " + patientsAtNAL.size)
    ev.Category match {
      case "RemovedPatient" => patientsAtNAL.remove(ev.CareContactId)
      case "Q" => patientsAtNAL.add(ev.CareContactId)
      case s: String if s.contains("removed") => patientsAtNAL.remove(ev.CareContactId)
      case _ => ()
    }
    ev.Type match {
      case "KLAR" => patientsAtNAL.remove(ev.CareContactId)
      case _ => ()
    }
  }
}