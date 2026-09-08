subroutine tsvc_2_s115_fp64(a, aa, len2d) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len2d
  real(c_double), intent(inout) :: a(len2d)
  real(c_double), intent(in) :: aa(len2d, len2d)

  integer, parameter :: B = 256
  integer(c_int64_t), parameter :: IC8 = 8
  integer(c_int64_t) :: i, j, jb, jlo, jhi, j2, t, nt, span, chunk, &
       ilo, ihi, i0, nlo, nhir, nb, tmax, base
  real(c_double) :: s
  real(c_double), allocatable :: tri(:)

  if (len2d < 512) then
     do j = 1, len2d - 1
        do i = j + 1, len2d
           a(i) = a(i) - aa(i, j) * a(j)
        end do
     end do
     return
  end if

  nt = omp_get_max_threads()
  nb = (len2d + B - 2) / B
  tmax = IC8 * B * (B + IC8)
  allocate (tri(0:tmax))

  ! prefill batch-0 band triangle
  do j2 = 2, min(len2d, int(B + 1, 8))
     do j = 1, j2 - 2
        tri((j2 - 1) * (j2 - 2) / 2 + (j - 1)) = aa(j2, j)
     end do
  end do

  !$omp parallel shared(a, aa, tri, nt, nb, len2d) private(jlo, jhi, j2, j, i, t, span, chunk, ilo, ihi, i0, nlo, nhir, jb, base, s)
  do jb = 0, nb - 1
     jlo = jb * B + 1
     jhi = min(jlo + B - 1, len2d - 1)
     ! band solve (serial): rows j2 = jlo+1 .. jhi+1
     !$omp single
        do j2 = jlo + 1, jhi + 1
           base = (j2 - jlo) * (j2 - jlo - 1) / 2
           s = 0.0d0
           do j = jlo, j2 - 2
              s = s + tri(base + (j - jlo)) * a(j)
           end do
           a(j2) = a(j2) - s - aa(j2, j2 - 1) * a(j2 - 1)
        end do
     !$omp end single
     ! fan-out (parallel over i-chunks)
     span = len2d - jhi - 1
     chunk = (span + nt - 1) / nt
     chunk = chunk + mod(IC8 - mod(chunk, IC8), IC8)
     !$omp do schedule(static)
     do t = 0, nt - 1
        ilo = jhi + 2 + t * chunk
        ihi = min(ilo + chunk - 1, len2d)
        if (ilo <= len2d) then
           do j = jlo, jhi
              i0 = max(ilo, j + 2)
              do i = i0, ihi
                 a(i) = a(i) - aa(i, j) * a(j)
              end do
           end do
        end if
     end do
     ! fill triangle for next batch (parallel)
     if (jb + 1 < nb) then
        nlo = jhi + 1
        nhir = min(nlo + B, len2d)
        !$omp do schedule(static)
        do j2 = nlo + 1, nhir
           do j = nlo, j2 - 2
              tri((j2 - nlo) * (j2 - nlo - 1) / 2 + (j - nlo)) = aa(j2, j)
           end do
        end do
     end if
  end do
  !$omp end parallel
end subroutine tsvc_2_s115_fp64
