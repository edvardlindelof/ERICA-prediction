package dataaggregation

import org.joda.time.DateTime
import sp.EricaEventLogger._
import sp.gPubSub.API_Data.EricaEvent

import scala.collection.mutable

// the output variable of QLasso article
// TODO haven't yet checked that the output of this one is reasonable
object TTLOfNextLowPrioPatient extends FutureTeller {
  override def futureState(events: List[EricaEvent]): List[(String, Int)] = {

    implicit def stringToDateTime(s: String) = DateTime.parse(s)

    val checkedPatients = mutable.Set[Int]()

    var toReturn = List[(String, Int)]()
    var done = false
    while(!done) {
      val nextPatEvent = events.find(ev => ev.Category == "Q" && !checkedPatients.contains(ev.CareContactId))
      if(!nextPatEvent.isDefined) {
        return toReturn
      }

      nextPatEvent.foreach { pEv =>
        val nextPatId = pEv.CareContactId
        checkedPatients.add(nextPatId)
        val prioEvent = events.find(ev => ev.CareContactId == nextPatId && ev.Category == "P")
        val firstDoctorEvent = events.find(ev => ev.CareContactId == nextPatId && ev.Type == "LÄKARE")
        prioEvent.foreach { prEv =>
          if (prEv.Type == "PRIO3" || prEv.Type == "PRIO4" || prEv.Type == "PRIO5")
            firstDoctorEvent.foreach { dEv =>
              val secondsToDoctor = (dEv.Start.getMillis - pEv.Start.getMillis) / 1000
              if (secondsToDoctor > 0) {
                done = true
                toReturn = List(("TTLOfNextPatient", secondsToDoctor.toInt))
              }
            }
        }
      }
    }

    return toReturn
  }
}
