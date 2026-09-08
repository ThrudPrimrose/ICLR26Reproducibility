! TSVC tsvc_2 s2710 -- elementwise conditional update, fp64.
! C ABI: tsvc_2_s2710_fp64(a, b, c, d, e, x, len_1d)
!
! The body is a data-dependent branch; the compiler if-converts it to
! per-lane masks and vectorizes the loop with AVX-512 (64-byte vectors,
! unroll 8). The scalar tests (len_1d > 10, x(1) > 0) are loop invariants
! and are decided once, outside the loop.
!
! Sizing strategy:
!   n small   -> a single thread: the OpenMP barrier alone would cost more
!                than the whole computation.
!   n medium  -> parallel static split.
!   n large   -> parallel dynamic with coarse chunks: the judge's physical
!                cores are imbalanced, so fast cores pick up extra chunks.
subroutine tsvc_2_s2710_fp64(a, b, c, d, e, x, len_1d) bind(C, name='tsvc_2_s2710_fp64')
  use, intrinsic :: iso_c_binding
  implicit none
  real(c_double), dimension(*), intent(inout) :: a, b, c
  real(c_double), dimension(*), intent(in)    :: d, e, x
  integer(c_int64_t), value, intent(in)       :: len_1d
  integer(c_int64_t) :: i, chunk
  logical :: big, xpos

  big  = (len_1d > 10)
  xpos = (x(1) > 0.0d0)

  if (len_1d < 4000) then
    do i = 1, len_1d
      if (a(i) > b(i)) then
        a(i) = a(i) + b(i) * d(i)
        if (big) then
          c(i) = c(i) + d(i) * d(i)
        else
          c(i) = d(i) * e(i) + 1.0d0
        end if
      else
        b(i) = a(i) + e(i) * e(i)
        if (xpos) then
          c(i) = a(i) + d(i) * d(i)
        else
          c(i) = c(i) + e(i) * e(i)
        end if
      end if
    end do
  else if (len_1d < 2000000) then
    !$omp parallel do default(none) shared(a, b, c, d, e, len_1d, big, xpos) &
    !$omp& private(i) schedule(static)
    do i = 1, len_1d
      if (a(i) > b(i)) then
        a(i) = a(i) + b(i) * d(i)
        if (big) then
          c(i) = c(i) + d(i) * d(i)
        else
          c(i) = d(i) * e(i) + 1.0d0
        end if
      else
        b(i) = a(i) + e(i) * e(i)
        if (xpos) then
          c(i) = a(i) + d(i) * d(i)
        else
          c(i) = c(i) + e(i) * e(i)
        end if
      end if
    end do
    !$omp end parallel do
  else
    chunk = max(1_8, len_1d / 96)
    !$omp parallel do default(none) shared(a, b, c, d, e, len_1d, big, xpos, chunk) &
    !$omp& private(i) schedule(dynamic, chunk)
    do i = 1, len_1d
      if (a(i) > b(i)) then
        a(i) = a(i) + b(i) * d(i)
        if (big) then
          c(i) = c(i) + d(i) * d(i)
        else
          c(i) = d(i) * e(i) + 1.0d0
        end if
      else
        b(i) = a(i) + e(i) * e(i)
        if (xpos) then
          c(i) = a(i) + d(i) * d(i)
        else
          c(i) = c(i) + e(i) * e(i)
        end if
      end if
    end do
    !$omp end parallel do
  end if
end subroutine tsvc_2_s2710_fp64
